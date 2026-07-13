"""市場バイアス地図: 「市場が複勝を過小評価する状況」を特定する（モデル非依存）。

全 place 候補(ER選別なし=ほぼ全出走馬)を feat_matrix に join し、状況(特徴)別に:
  - bet_all_ROI = その状況で複勝を全部¥100買ったときのROI。>0.80(控除床)超=市場が過小評価=機会。
  - actual = 実複勝率、model_p = モデル予測複勝確率の平均、ratio = model_p/actual(>1=過信)。
market が構造的に外す(bet_all_ROI高)状況を探し、そこで我々のモデルが実績に近い(ratio≈1)なら
＝(市場ミスプライス)∩(我々に予測力)＝集中すべきエッジ帯。

使い方(hro-backtest, HRO_ABLATE_SED は学習時と同じ):
    HRO_ABLATE_SED=1 poetry run python scripts/market_bias.py [wf_cand_dir]
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

# 市場が情報薄=予測困難と思しき状況を優先的に見る + 出目に効く構造
NUMERIC = ["days_since_prev", "class_ratio", "field_size", "distance_m",
           "jrdb_oikire_index", "jrdb_shiagari_index", "jrdbk_gekiso",
           "jrdbk_idm", "jrdbk_kijun_ninki", "kab_baba_sa",
           "h_speedidx_adj_6m", "earnings_cum", "h_fukusyo_6m"]
CATEG = ["surface", "baba_state", "grade_cd", "syubetu_cd",
         "jrdbk_kyakushitsu", "jrdb_cyokyo_ryo", "jrdb_cyokyo_eval",
         "flag_no_history", "flag_missing_prev"]  # 新馬/前走なし=市場情報薄


def _load_all_place(cand_dir):
    """(race_id, umaban, odds, payout, model_p, hit) 全place候補(ER選別なし)。"""
    out = []
    for f in glob.glob(os.path.join(cand_dir, "place_*.csv")):
        with open(f, encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for x in r:
                if not x or x[0] != "place" or x[4] != "True":
                    continue
                rid = x[9] if len(x) > 9 else ""
                sel = x[10] if len(x) > 10 else ""
                if not rid or not sel.strip().isdigit():
                    continue
                out.append((rid, int(sel), float(x[3]), int(x[6]),
                            float(x[2]), 1 if x[5] == "True" else 0))
    return out


def _stats(sub):
    """(n, bet_all_ROI, actual_rate, model_p_mean, ratio)"""
    n = len(sub)
    if not n:
        return (0, float("nan"), float("nan"), float("nan"), float("nan"))
    roi = sum(pay for _o, pay, _p, _h in sub) / (n * 100)
    act = sum(h for _o, _pay, _p, h in sub) / n
    mp = sum(p for _o, _pay, p, _h in sub) / n
    ratio = mp / act if act else float("nan")
    return (n, roi, act, mp, ratio)


def _flag(roi):
    if roi >= 0.95:
        return "  << 市場過小評価(機会)"
    if roi <= 0.70:
        return "  (市場適正〜過大)"
    return ""


def main():
    cand = _load_all_place(CAND_DIR)
    if not cand:
        print(f"{CAND_DIR} に馬番入り候補なし。enriched で再collectを。")
        return 1
    n, roi, act, mp, ratio = _stats([(o, pay, p, h) for _r, _u, o, pay, p, h in cand])
    print(f"全place候補: n={n} bet_all_ROI={roi:.4f} 実複勝率={act:.3f} "
          f"モデルp平均={mp:.3f} (過信ratio={ratio:.2f})\n")

    ids = sorted({c[0] for c in cand})
    cols = NUMERIC + CATEG
    db = FeatureDB(load_config())
    sql = ("SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) AS rid, umaban, "
           + ", ".join(cols)
           + " FROM feat_matrix WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num)=ANY(%(ids)s)")
    feat = {}
    for row in db.query(sql, {"ids": ids}):
        try:
            feat[(row["rid"], int(row["umaban"]))] = row
        except (TypeError, ValueError):
            continue
    db.close()

    data = []  # (odds, payout, model_p, hit, featrow)
    for rid, uma, o, pay, p, h in cand:
        fr = feat.get((rid, uma))
        if fr is not None:
            data.append((o, pay, p, h, fr))

    def rowstats(sub):
        return _stats([(o, pay, p, h) for o, pay, p, h, _f in sub])

    def show_num(col):
        vals = [float(fr[col]) for *_x, fr in data if fr.get(col) is not None]
        if len(vals) < 200:
            print(f"[{col}] n<200 skip")
            return
        xs = sorted(vals)
        q = [xs[int(len(xs) * k / 4)] for k in (1, 2, 3)]

        def b(v):
            v = float(v)
            return 0 if v <= q[0] else 1 if v <= q[1] else 2 if v <= q[2] else 3
        g = defaultdict(list)
        for o, pay, p, h, fr in data:
            if fr.get(col) is not None:
                g[b(fr[col])].append((o, pay, p, h, fr))
        print(f"[{col}]  四分位: n / bet_all_ROI / 実率 / モデルp / 過信ratio")
        for i in range(4):
            n, roi, act, mp, ratio = rowstats(g[i])
            lab = (f"<={q[0]:.3g}" if i == 0 else f">{q[2]:.3g}" if i == 3
                   else f"{q[i-1]:.3g}~{q[i]:.3g}")
            print(f"  Q{i+1} {lab:>15} n={n:5} ROI={roi:.3f} act={act:.3f} "
                  f"mdl={mp:.3f} r={ratio:.2f}{_flag(roi)}")
        print()

    def show_cat(col):
        g = defaultdict(list)
        for row in data:
            g[str(row[4].get(col))].append(row)
        items = sorted(g.items(), key=lambda kv: -len(kv[1]))[:8]
        print(f"[{col}]  値: n / bet_all_ROI / 実率 / モデルp / 過信ratio")
        for v, sub in items:
            n, roi, act, mp, ratio = rowstats(sub)
            print(f"  {v:>8} n={n:5} ROI={roi:.3f} act={act:.3f} "
                  f"mdl={mp:.3f} r={ratio:.2f}{_flag(roi)}")
        print()

    for c in NUMERIC:
        show_num(c)
    for c in CATEG:
        show_cat(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
