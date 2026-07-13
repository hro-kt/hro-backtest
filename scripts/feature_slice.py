"""複勝ベットの ROI を主要特徴量でスライスし、モデルの得手/不得手マップを作る。

wf_cand/place_*.csv(馬番 selection_id 入り。enriched で再collect したもの) を feat_matrix に
race_id+umaban で join し、重要特徴ごとに ROI/hit% を出す。ROI<1 の特徴領域＝モデルが外す所＝
次に作る/直す特徴の狙い所。odds/休養のような表層でなく「モデルが使う特徴」で切るのが王道。

使い方(hro-backtest ディレクトリ, HRO_ABLATE_SED は学習時と同じに):
    HRO_ABLATE_SED=1 poetry run python scripts/feature_slice.py [wf_cand_dir]
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

from hro_features.config import load_config
from hro_features.db import FeatureDB

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/hro/wf_cand"

# モデルが使う重要特徴(数値)。収得賞金/過去走複勝/クラス変化/スピード/CYB調教/KYI/KAB/展開 等。
NUMERIC = [
    "earnings_per_start", "earnings_cum", "race_class", "class_ratio",
    "h_fukusyo_1y", "h_fukusyo_6m", "h_win_1y", "h_finish_sd_1y",
    "h_speedidx_6m", "h_speedidx_adj_6m", "field_size", "distance_m",
    "days_since_prev", "prev_agari_rank",
    "jrdb_oikire_index", "jrdb_shiagari_index",
    "jrdbk_gekiso", "jrdbk_idm", "jrdbk_josho", "jrdbk_kijun_ninki",
    "kab_baba_sa",
]
# カテゴリ特徴。
CATEG = ["surface", "baba_state", "grade_cd", "syubetu_cd",
         "jrdb_cyokyo_eval", "jrdbk_kyakushitsu"]


def _load_bets(cand_dir: str) -> list[tuple[str, int, int]]:
    """(race_id, umaban, payout) の複勝ベット(er>=1.3 & prob>=0.15 & settled)。"""
    bets: list[tuple[str, int, int]] = []
    for f in glob.glob(os.path.join(cand_dir, "place_*.csv")):
        with open(f, encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for x in r:
                if not x or x[0] != "place":
                    continue
                if float(x[1]) < 1.3 or float(x[2]) < 0.15 or x[4] != "True":
                    continue
                rid = x[9] if len(x) > 9 else ""
                sel = x[10] if len(x) > 10 else ""
                if not rid or not sel.strip().isdigit():
                    continue
                bets.append((rid, int(sel), int(x[6])))
    return bets


def _roi(sub: list[int]) -> tuple[int, float, float]:
    n = len(sub)
    if not n:
        return (0, float("nan"), 0.0)
    return (n, sum(sub) / (n * 100), 100 * sum(1 for p in sub if p > 0) / n)


def _flag(roi: float) -> str:
    if roi < 0.95:
        return "  << 弱(外す領域)"
    if roi > 1.12:
        return "  >> 強"
    return ""


def main() -> int:
    bets = _load_bets(CAND_DIR)
    if not bets:
        print(f"{CAND_DIR} に馬番入り候補がありません。enriched で再collect しましたか？")
        return 1
    base_n = len(bets)
    base_roi = sum(b[2] for b in bets) / (base_n * 100)
    print(f"bets={base_n}  baseline ROI={base_roi:.4f}")

    ids = sorted({b[0] for b in bets})
    cols = NUMERIC + CATEG
    db = FeatureDB(load_config())
    sql = (
        "SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) AS rid, umaban, "
        + ", ".join(cols)
        + " FROM feat_matrix "
        "WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num) = ANY(%(ids)s)"
    )
    feat: dict[tuple[str, int], dict] = {}
    for row in db.query(sql, {"ids": ids}):
        try:
            feat[(row["rid"], int(row["umaban"]))] = row
        except (TypeError, ValueError):
            continue
    db.close()

    data: list[tuple[int, dict]] = []
    miss = 0
    for rid, uma, pay in bets:
        fr = feat.get((rid, uma))
        if fr is None:
            miss += 1
            continue
        data.append((pay, fr))
    print(f"joined={len(data)} miss={miss}\n")

    def show_numeric(col: str) -> None:
        vals = [fr[col] for _p, fr in data if fr.get(col) is not None]
        if len(vals) < 60:
            print(f"[{col}] n<60 skip")
            return
        xs = sorted(float(v) for v in vals)
        q = [xs[int(len(xs) * k / 4)] for k in (1, 2, 3)]

        def bucket(v: float) -> int:
            v = float(v)
            return 0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3

        groups: dict[int, list[int]] = defaultdict(list)
        for p, fr in data:
            v = fr.get(col)
            if v is None:
                continue
            groups[bucket(v)].append(p)
        print(f"[{col}]  四分位: n / ROI / hit%")
        for i in range(4):
            n, r, h = _roi(groups[i])
            lab = (f"<={q[0]:.3g}" if i == 0
                   else f">{q[2]:.3g}" if i == 3
                   else f"{q[i-1]:.3g}~{q[i]:.3g}")
            print(f"  Q{i+1} {lab:>16} n={n:4} ROI={r:.3f} hit={h:4.1f}%{_flag(r)}")
        print()

    def show_categ(col: str) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for p, fr in data:
            groups[str(fr.get(col))].append(p)
        items = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
        print(f"[{col}]  値: n / ROI / hit%")
        for v, sub in items:
            n, r, h = _roi(sub)
            print(f"  {v:>10} n={n:4} ROI={r:.3f} hit={h:4.1f}%{_flag(r)}")
        print()

    for c in NUMERIC:
        show_numeric(c)
    for c in CATEG:
        show_categ(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
