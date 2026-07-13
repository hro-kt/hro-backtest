"""winner's curse 補正(信頼度別 shrinkage)を walk-forward OOS でオフライン検証する。

ER≥1.3 選別は「p_model が高い側=正の推定誤差」を拾う→実現ROIが下振れ(winner's curse)。
そこで **推定が不確実な馬(過去走 h_n_2y が少ない)ほど p_model を市場implied側へ縮小**する:
    λ_i = K/(K + h_n_2y_i)          # 経験が少ない=不確実→λ大→市場へ強く寄せる
    p_market ≈ 0.8 / place_odds     # 複勝の市場implied確率の近似(控除~20%)
    p_shrunk = (1-λ_i)·p_model + λ_i·p_market
    ER_shrunk = p_shrunk · odds  (= (1-λ)·ER_raw + 0.8λ)
ER_shrunk≥1.3 & p_shrunk≥0.15 で選別。ノイズで膨れたbetが落ち、確信度の高い乖離だけ残る想定。
K=0 が baseline(縮小なし)。K を数点、年別+プールで比較。全年でbaseline超なら採用候補。

h_n_2y は候補CSVの seg_runs(列8)にあるので DB join 不要。
使い方: poetry run python scripts/shrink_test.py [wf_cand_dir]
"""

from __future__ import annotations

import csv
import glob
import os
import sys

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/hro/wf_cand"
MIN_ER, MIN_PROB = 1.3, 0.15
K_LIST = [0, 2, 5, 10, 20]


def _load_by_year(cand_dir):
    by = {}
    for fp in sorted(glob.glob(os.path.join(cand_dir, "place_*.csv"))):
        y = os.path.basename(fp)[6:-4]
        rows = []
        with open(fp, encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for x in r:
                if not x or x[0] != "place" or x[4] != "True":
                    continue
                prob, odds, pay = float(x[2]), float(x[3]), int(x[6])
                hit = 1 if x[5] == "True" else 0
                sr = x[7] if len(x) > 7 else ""
                n2y = int(sr) if sr.strip().lstrip("-").isdigit() else 0
                rows.append((prob, odds, pay, hit, max(n2y, 0)))
        by[y] = rows
    return by


def _select_roi(rows, K):
    """K で shrinkage 選別した (n, ROI)。"""
    sel_pay = sel_n = 0
    for prob, odds, pay, _hit, n2y in rows:
        lam = 0.0 if K == 0 else K / (K + n2y)
        p_mkt = 0.8 / odds
        p_shr = (1 - lam) * prob + lam * p_mkt
        er = p_shr * odds
        if er >= MIN_ER and p_shr >= MIN_PROB:
            sel_n += 1
            sel_pay += pay
    roi = sel_pay / (sel_n * 100) if sel_n else float("nan")
    return sel_n, roi


def main():
    by = _load_by_year(CAND_DIR)
    if not by:
        print(f"{CAND_DIR} に候補なし")
        return 1
    years = list(by)
    allrows = [r for rs in by.values() for r in rs]
    hdr = "K(shrink)".ljust(10) + " | " + " | ".join(y.rjust(12) for y in years) + " |        ALL"
    print(hdr)
    print("-" * len(hdr))
    for K in K_LIST:
        line = (f"K={K}" if K else "K=0(base)").ljust(10)
        for y in years:
            n, roi = _select_roi(by[y], K)
            line += f" | {roi:.3f}({n:>4})"
        n, roi = _select_roi(allrows, K)
        line += f" | {roi:.3f}({n})"
        print(line)
    print("\n全年でbaseline(K=0)以上のKがあれば winner's curse 補正が有効。無ければ限界確定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
