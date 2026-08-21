"""予測ログのバックフィル: 過去JRAレースをモデルで採点し prediction_log へ保存。

MLOps 監視(較正/ドリフト/識別)の土台を過去9年分いっきに用意する。
効率のため feat_matrix(MV, 索引付) を レースキー順に **一括ストリーム** し、連続する
同一レースの行をまとめて score_abilities で採点する(online の per-race view 再計算を避ける)。
前向き(race_day)と同じ prediction_log スキーマ・同じ score_abilities を使うので整合。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB
from hro_features.spec import FEATURE_COLUMNS, KEY_COLUMNS, feature_schema_hash

from hro_predictor.predict import score_abilities

from . import harness

_JST = timezone(timedelta(hours=9))
_JRA = ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
_RACE_KEYS = ("year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num")

# feat_matrix から採点に必要な列(キー+特徴)。ラベルは読まない(較正時に別途 join)。
_SELECT_COLS = ", ".join(dict.fromkeys(KEY_COLUMNS + FEATURE_COLUMNS))  # 重複排除して順序保持

_UPSERT = """
INSERT INTO prediction_log
    (race_id, year, month_day, jyo_cd, kaiji, nichiji, race_num, umaban, selection_id,
     p_win, p_place, field_size, odds_tan, ninki_tan, model_version, source, decided_at)
VALUES
    (%(race_id)s, %(year)s, %(month_day)s, %(jyo_cd)s, %(kaiji)s, %(nichiji)s, %(race_num)s,
     %(umaban)s, %(selection_id)s, %(p_win)s, %(p_place)s, %(field_size)s,
     %(odds_tan)s, %(ninki_tan)s, %(model_version)s, %(source)s, %(decided_at)s)
ON CONFLICT (year, month_day, jyo_cd, kaiji, nichiji, race_num, umaban, model_version, source)
DO UPDATE SET
    p_win=EXCLUDED.p_win, p_place=EXCLUDED.p_place, field_size=EXCLUDED.field_size,
    odds_tan=EXCLUDED.odds_tan, ninki_tan=EXCLUDED.ninki_tan,
    selection_id=EXCLUDED.selection_id, decided_at=EXCLUDED.decided_at
"""


def model_version_of(meta) -> str:
    """モデル世代の識別子。特徴schemaハッシュ+学習終端で世代を区別。"""
    end = meta.train_range[1] if getattr(meta, "train_range", None) else "?"
    return f"{meta.schema_hash[:8]}@{end}"


def _decided_at(year: str, month_day: str) -> datetime:
    """backfill の予測時刻はレース日(JST 0時)。時刻バケット/最新性の基準に使う。"""
    try:
        return datetime(int(year), int(month_day[:2]), int(month_day[2:]), tzinfo=_JST)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_pred_rows(race, abilities: dict, model_version: str, source: str,
                    decided_at, feat_by_umaban: dict | None = None) -> list[dict]:
    """abilities(score_abilities の戻り) → prediction_log 行の dict リスト。

    race は6要素キー(year,month_day,jyo_cd,kaiji,nichiji,race_num)。feat_by_umaban があれば
    as-of オッズ(odds_tan/ninki_tan)を付与する(無ければ NULL)。backfill/前向き 共通。
    """
    feat_by_umaban = feat_by_umaban or {}
    race_id = "".join(race)
    rows = []
    for rn in abilities.get("runners", []):
        umaban = str(rn.get("umaban"))
        src = feat_by_umaban.get(umaban, {})
        rows.append({
            "race_id": race_id, "year": race[0], "month_day": race[1],
            "jyo_cd": race[2], "kaiji": race[3], "nichiji": race[4], "race_num": race[5],
            "umaban": umaban, "selection_id": rn.get("selection_id"),
            "p_win": _num(rn.get("p_win")), "p_place": _num(rn.get("p_place")),
            "field_size": abilities.get("field_size"),
            "odds_tan": _num(src.get("odds_tan")), "ninki_tan": _int(src.get("ninki_tan")),
            "model_version": model_version, "source": source, "decided_at": decided_at,
        })
    return rows


def upsert_predictions(conn, rows: list[dict], *, commit: bool = True) -> int:
    """prediction_log へ UPSERT。conn は psycopg 接続。書いた行数を返す。"""
    if not rows:
        return 0
    with conn.cursor() as c:
        c.executemany(_UPSERT, rows)
    if commit:
        conn.commit()
    return len(rows)


def backfill_predictions(
    d_from: str, d_to: str, win_path: str, place_path: str, *,
    source: str = "backfill", limit: int | None = None, show_progress: bool = True,
    batch: int = 2000,
) -> dict:
    """[d_from,d_to] のJRAレースを採点し prediction_log へ UPSERT。件数サマリを返す。"""
    win_b, place_b = harness.load_models(win_path, place_path)
    mv = model_version_of(win_b.meta)
    cfg = load_features_config()
    db = FeatureDB(cfg)

    import psycopg
    wconn = psycopg.connect(cfg.conninfo, autocommit=False)

    sql = (
        f"SELECT {_SELECT_COLS} FROM feat_matrix "
        f"WHERE (year || month_day) BETWEEN %(a)s AND %(b)s "
        f"  AND jyo_cd IN {_JRA} "
        f"ORDER BY year, month_day, jyo_cd, kaiji, nichiji, race_num, umaban"
    )

    n_races = n_rows = 0
    pending: list[dict] = []
    cur_key = None
    group: list[dict] = []

    def flush_batch():
        if not pending:
            return
        upsert_predictions(wconn, pending, commit=True)
        pending.clear()

    def score_group(rows: list[dict]):
        nonlocal n_races, n_rows
        if not rows:
            return
        race = tuple(rows[0][k] for k in _RACE_KEYS)
        race_id = "".join(race)
        abil = score_abilities(rows, win_b, place_b, race_id)
        by_umaban = {str(r.get("umaban")): r for r in rows}
        prows = build_pred_rows(race, abil, mv, source, _decided_at(race[0], race[1]),
                                feat_by_umaban=by_umaban)
        pending.extend(prows)
        n_rows += len(prows)
        n_races += 1

    bar = harness._progress_bar(0, show_progress) if hasattr(harness, "_progress_bar") else None
    try:
        for chunk in db.stream(sql, {"a": d_from, "b": d_to}, chunk=batch):
            for row in chunk:
                key = tuple(row[k] for k in _RACE_KEYS)
                if cur_key is not None and key != cur_key:
                    score_group(group)
                    group = []
                    if bar:
                        bar.update(1)
                    if len(pending) >= batch:
                        flush_batch()
                    if limit and n_races >= limit:
                        cur_key = None
                        group = []
                        break
                cur_key = key
                group.append(row)
            else:
                continue
            break  # limit 到達
        score_group(group)  # 最後のレース
        flush_batch()
    finally:
        if bar:
            bar.close()
        wconn.close()
        db.close()

    return {"model_version": mv, "source": source, "races": n_races,
            "rows": n_rows, "window": f"{d_from}..{d_to}"}
