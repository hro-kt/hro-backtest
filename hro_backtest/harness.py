"""E2E ハーネス: 1レース駆動と期間バックテスト。

データフロー（学習・予測と同一の feat_matrix を使用）:
  build_race_features(online) → score_abilities(win/place モデル) → decide_race(PL)
  → run_decide_pipeline(資金配分=BetOrder) → settlement(nl_hr 払戻突合) → ROI/的中率

接続は3系統（row_factory が異なるため分離）:
  - FeatureDB          : 特徴(online)・レース一覧。dict_row。
  - optimizer.connect  : オッズ load_odds_lookup。dict_row。
  - psycopg(tuple rows): nl_hr load_payout_rows（タプル展開で読むため既定 row factory）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB
from hro_features.online import build_race_features
from hro_features.spec import FEATURE_COLUMNS, feature_schema_hash

from hro_predictor.bundle import ModelBundle
from hro_predictor.predict import score_abilities, TARGET_PLACE, TARGET_WIN

from hro_optimizer.config import BettingConfig, KellyConfig, SimConfig
from hro_optimizer.db import PostgresConfig, connect as opt_connect, load_odds_lookup
from hro_optimizer.engine import decide_race
from hro_optimizer.io import race_abilities_from_dict

from hro_moneymanager.config import MoneyManagerConfig
from hro_moneymanager.pipeline import run_decide_pipeline

from hro_buyer.models import STATUS_DRY_RUN, ExecutionResult
from hro_buyer.postgres import load_payout_rows
from hro_buyer.settlement import build_payout_index, settle_result, settle_results, summarize

RaceKey = tuple[str, str, str, str, str, str]


def betting_config(source: str, *, min_er=None, min_prob=None, max_odds_age=None) -> BettingConfig:
    """confirmed は鮮度ガード無制限を既定に（確定オッズは鮮度概念なし）。閾値は任意で上書き。"""
    d = BettingConfig()
    age = max_odds_age
    if age is None:
        age = float("inf") if source == "confirmed" else d.max_odds_age_seconds
    return BettingConfig(
        min_expected_return=(min_er if min_er is not None else d.min_expected_return),
        min_probability=(min_prob if min_prob is not None else d.min_probability),
        max_odds_age_seconds=age,
    )


def load_models(win_path: str, place_path: str) -> tuple[ModelBundle, ModelBundle]:
    """win/place バンドルを読み、現スキーマ一致と役割(target)を検証して返す。"""
    cur_hash = feature_schema_hash()
    win_b = ModelBundle.load(win_path)
    place_b = ModelBundle.load(place_path)
    win_b.meta.assert_compatible(cur_hash, FEATURE_COLUMNS)
    place_b.meta.assert_compatible(cur_hash, FEATURE_COLUMNS)
    if win_b.meta.target != TARGET_WIN:
        raise ValueError(f"--win-model の target は {TARGET_WIN} であるべき: {win_b.meta.target}")
    if place_b.meta.target != TARGET_PLACE:
        raise ValueError(f"--place-model の target は {TARGET_PLACE} であるべき: {place_b.meta.target}")
    return win_b, place_b


def list_races(db: FeatureDB, d_from: str, d_to: str, limit: int | None = None) -> list[RaceKey]:
    """[d_from, d_to](YYYYMMDD) の確定レース(結果あり)のレースキー一覧。"""
    rows = db.query(
        "SELECT DISTINCT year, month_day, jyo_cd, kaiji, nichiji, race_num "
        "FROM feat_labels WHERE (year || month_day) BETWEEN %(a)s AND %(b)s "
        "ORDER BY year, month_day, jyo_cd, race_num",
        {"a": d_from, "b": d_to},
    )
    races = [(r["year"], r["month_day"], r["jyo_cd"], r["kaiji"], r["nichiji"], r["race_num"])
             for r in rows]
    return races[:limit] if limit else races


def orders_for_race(
    db: FeatureDB, conn_odds, win_b: ModelBundle, place_b: ModelBundle, race: RaceKey, *,
    betting: BettingConfig, money: MoneyManagerConfig, sim: SimConfig, kelly: KellyConfig,
    source: str, simultaneous: bool,
) -> tuple[dict | None, list]:
    """1レース: 能力値→PL候補→資金配分。戻り (abilities_dict|None, BetOrder list)。"""
    rows = build_race_features(db, *race)
    if not rows:
        return None, []
    race_id = "".join(race)
    abilities_dict = score_abilities(rows, win_b, place_b, race_id)
    abilities = race_abilities_from_dict(abilities_dict)
    odds_lookup = load_odds_lookup(conn_odds, race, source=source)
    result = run_decide_pipeline(
        abilities, odds_lookup, betting, money,
        sim_config=sim, kelly_config=kelly, simultaneous=simultaneous,
    )
    return abilities_dict, result.orders


def _to_exec(order) -> ExecutionResult:
    """BetOrder → settlement が突合できる ExecutionResult(dry_run 記録扱い)。"""
    return ExecutionResult(
        race_id=order.race_id, selection_id=order.selection_id, bet_type=order.bet_type,
        amount=order.amount, odds=order.odds, mode="dry_run", status=STATUS_DRY_RUN,
        message="backtest",
    )


@dataclass
class BacktestResult:
    n_races: int
    n_races_with_orders: int
    summary: object                  # SettlementSummary（全体）
    by_bet_type: dict                # {bet_type: SettlementSummary}
    settlements: list                # 個別 Settlement（詳細出力・CSV 用）


def _progress_bar(total: int, show: bool):
    """tqdm があれば進捗バー、無ければ None（呼び出し側が periodic ログにフォールバック）。"""
    if not show:
        return None
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm(total=total, unit="race", desc="backtest", file=sys.stderr)


def run_backtest(
    d_from: str, d_to: str, win_path: str, place_path: str, *,
    source: str = "confirmed", samples: int | None = None, max_total: float | None = None,
    independent_kelly: bool = False, min_er=None, min_prob=None, max_odds_age=None,
    limit: int | None = None, show_progress: bool = True,
) -> BacktestResult:
    """期間バックテスト: 各レースで bet を生成し、nl_hr 払戻と突合して ROI を出す。

    資金配分はレース独立(store なし)。ROI は比率なので予算スケールに非依存。
    show_progress=True で tqdm 進捗バー(未導入時は 100 レース毎に stderr へログ)。
    """
    win_b, place_b = load_models(win_path, place_path)
    betting = betting_config(source, min_er=min_er, min_prob=min_prob, max_odds_age=max_odds_age)
    money = MoneyManagerConfig()
    sim = SimConfig(n_samples=samples) if samples else SimConfig()
    kelly = KellyConfig(max_total=max_total) if max_total else KellyConfig()

    # nl_hr は tuple-row で読む（load_payout_rows がタプル展開のため）別接続を保持
    import psycopg
    db = FeatureDB(load_features_config())
    conn_odds = opt_connect()
    conn_hr = psycopg.connect(PostgresConfig.from_env().conninfo)
    n_with = 0
    settlements: list = []
    run = [0, 0, 0]  # staked, payout, n_settled（走行中ROI表示用）
    races = list_races(db, d_from, d_to, limit=limit)
    n = len(races)
    bar = _progress_bar(n, show_progress)
    try:
        for i, race in enumerate(races, 1):
            _, orders = orders_for_race(
                db, conn_odds, win_b, place_b, race,
                betting=betting, money=money, sim=sim, kelly=kelly,
                source=source, simultaneous=not independent_kelly,
            )
            if orders:
                n_with += 1
                # 1レースぶんを即突合 → 走行中ROIを更新（途中経過が見える）
                idx, settled = build_payout_index(load_payout_rows(conn_hr, {orders[0].race_id}))
                sts = settle_results([_to_exec(o) for o in orders], idx, settled)
                settlements.extend(sts)
                for s in sts:
                    if s.settled:
                        run[0] += s.amount
                        run[1] += s.payout
                        run[2] += 1
            if bar is not None:
                bar.update(1)
                roi = (run[1] / run[0]) if run[0] else None
                bar.set_postfix(bets=len(settlements), w_orders=n_with,
                                ROI=("--" if roi is None else f"{roi:.3f}"),
                                pnl=run[1] - run[0])
            elif show_progress and (i % 100 == 0 or i == n):
                roi = (run[1] / run[0]) if run[0] else float("nan")
                print(f"  ... {i}/{n} races bets={len(settlements)} "
                      f"ROI={roi:.3f} pnl={run[1] - run[0]:+d}", file=sys.stderr)
    finally:
        if bar is not None:
            bar.close()
        conn_hr.close()
        conn_odds.close()
        db.close()

    by_type = {}
    for bt in ("place", "wide", "trio", "win"):
        sub = [s for s in settlements if s.bet_type == bt]
        if sub:
            by_type[bt] = summarize(sub)

    return BacktestResult(
        n_races=len(races), n_races_with_orders=n_with,
        summary=summarize(settlements), by_bet_type=by_type, settlements=settlements,
    )


# --------------------------------------------------------------------------- #
# 閾値スイープ: 全候補を1パスで評価＆突合 → グリッドはメモリ集計（フラット100円ベット）
# --------------------------------------------------------------------------- #
def collect_settled_candidates(
    d_from: str, d_to: str, win_path: str, place_path: str, *,
    source: str = "confirmed", samples: int | None = None,
    limit: int | None = None, show_progress: bool = True,
    bet_types: tuple[str, ...] | None = None,
) -> list[tuple]:
    """全レースの「フィルタ前の全候補」を評価し、nl_hr で突合した結果を返す。

    返り: list[(bet_type, expected_return, probability, settled, hit, payout_per_100)]。
    閾値(min_er/min_prob)に依らない重い処理(特徴/PL確率/突合)はここで1回だけ実施し、
    スイープはこのリストの filter+集計で行う（パイプラインを設定ごとに回さない）。
    フラット100円ベット前提（payout_per_100 がそのまま的中時の払戻）。

    bet_types を絞ると評価対象の券種が減る（trio は C(頭数,3) で候補・PL MC の大半を
    占めるため、不要なら外すと eval パスが大幅に速く・省メモリになる）。
    """
    win_b, place_b = load_models(win_path, place_path)
    d = BettingConfig()
    age = float("inf") if source == "confirmed" else d.max_odds_age_seconds
    # しきい値は無効化（min_er/min_prob=0）。odds 範囲は通常ポリシー(1.5..50)を維持。
    bt_kw = {"allowed_bet_types": tuple(bet_types)} if bet_types else {}
    permissive = BettingConfig(min_expected_return=0.0, min_probability=0.0,
                               max_odds_age_seconds=age, **bt_kw)
    sim = SimConfig(n_samples=samples) if samples else SimConfig()
    kelly = KellyConfig()

    # nl_hr は tuple-row で読む別接続を保持し、1レースぶんを即突合（走行中ROIを表示）
    import psycopg
    db = FeatureDB(load_features_config())
    conn_odds = opt_connect()
    conn_hr = psycopg.connect(PostgresConfig.from_env().conninfo)
    out: list[tuple] = []   # (bet_type, er, prob, settled, hit, payout_per_100)
    run = [0, 0]            # staked, payout（全候補フラット¥100の走行中ROI）
    races = list_races(db, d_from, d_to, limit=limit)
    bar = _progress_bar(len(races), show_progress)
    try:
        for i, race in enumerate(races, 1):
            rows = build_race_features(db, *race)
            if rows:
                race_id = "".join(race)
                ab = race_abilities_from_dict(score_abilities(rows, win_b, place_b, race_id))
                odds_lookup = load_odds_lookup(conn_odds, race, source=source)
                res = decide_race(ab, odds_lookup, permissive,
                                  sim_config=sim, kelly_config=kelly, simultaneous=False)
                if res.candidates:
                    idx, settled_races = build_payout_index(load_payout_rows(conn_hr, {race_id}))
                    for c in res.candidates:
                        s = settle_result(
                            ExecutionResult(race_id=c.race_id, selection_id=c.selection_id,
                                            bet_type=c.bet_type, amount=100, odds=c.odds,
                                            mode="dry_run", status=STATUS_DRY_RUN, message="sweep"),
                            idx, settled_races,
                        )
                        out.append((c.bet_type, c.expected_return, c.probability,
                                    s.settled, s.hit, s.payout))
                        if s.settled:
                            run[0] += 100
                            run[1] += s.payout
            if bar is not None:
                bar.update(1)
                roi = (run[1] / run[0]) if run[0] else None
                bar.set_postfix(cands=len(out), ROI=("--" if roi is None else f"{roi:.3f}"))
            elif show_progress and (i % 100 == 0 or i == len(races)):
                roi = (run[1] / run[0]) if run[0] else float("nan")
                print(f"  ... {i}/{len(races)} races cands={len(out)} ROI(all)={roi:.3f}",
                      file=sys.stderr)
    finally:
        if bar is not None:
            bar.close()
        conn_hr.close()
        conn_odds.close()
        db.close()
    return out


def sweep_roi(
    settled: list[tuple], er_grid: list[float], prob_grid: list[float],
) -> dict[str, dict[tuple[float, float], tuple[int, float | None, float | None]]]:
    """評価済み候補から (min_er, min_prob) ごとの ROI を集計（フラット100円ベット）。

    返り: {bet_type or 'ALL': {(er, prob): (n, roi, hit_rate)}}。
    roi = 払戻合計 / 投資合計、hit_rate = 的中 / ベット数（確定分のみ）。
    """
    types = ["ALL", "place", "wide", "trio"]
    table: dict[str, dict] = {t: {} for t in types}
    for er in er_grid:
        for pr in prob_grid:
            acc = {t: [0, 0, 0, 0] for t in types}  # n, stake, payout, hits
            for bt, e, p, st, hit, pay in settled:
                if not st or e < er or p < pr:
                    continue
                for key in ("ALL", bt):
                    a = acc.get(key)
                    if a is None:
                        continue
                    a[0] += 1
                    a[1] += 100
                    a[2] += pay
                    a[3] += 1 if hit else 0
            for t in types:
                n, stake, payout, hits = acc[t]
                roi = (payout / stake) if stake else None
                hr = (hits / n) if n else None
                table[t][(er, pr)] = (n, roi, hr)
    return table

