"""モデル陳腐化(staleness)検証: 学習データから離れるほど複勝ROIが落ちるか。

walk-forward は year Y を「1モデル」で予測(train末=前年6月, test=Y年1-12月)。
＝Y年1月=約7ヶ月古い, 12月=約18ヶ月古い。もし年内で1月→12月にROIが劣化するなら、
毎月/毎週再学習で鮮度を保てば性能向上が期待できる(=annual評価は過小評価)。

既存 wf_cand(候補にrace_id=日付)から、月別(=陳腐化の代理)にROIを集計。全9年プール。
使い方: poetry run python scripts/staleness.py [wf_cand_dir]
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/hro/wf_cand"


def _roi(pays):
    n = len(pays)
    return (n, sum(pays) / (n * 100) if n else float("nan"),
            100 * sum(1 for p in pays if p > 0) / n if n else 0)


def main():
    by_month = defaultdict(list)          # 月(1-12) -> payout list (全年プール)
    by_month_year = defaultdict(list)     # (year,月) -> payout (年内トレンド用)
    for f in glob.glob(os.path.join(CAND_DIR, "place_*.csv")):
        with open(f, encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for x in r:
                if not x or x[0] != "place" or x[4] != "True":
                    continue
                if float(x[1]) < 1.3 or float(x[2]) < 0.15:
                    continue
                rid = x[9] if len(x) > 9 else ""
                if len(rid) < 8:
                    continue
                mm = int(rid[4:6])
                yy = rid[0:4]
                by_month[mm].append(int(x[6]))
                by_month_year[(yy, mm)].append(int(x[6]))

    print("=== 月別ROI(全9年プール) 学習データ末=各year前年6月 → 月が進むほど陳腐化 ===")
    print(" 月 | 陳腐化(概算) |    n |  ROI  | hit%")
    for m in range(1, 13):
        n, roi, hit = _roi(by_month.get(m, []))
        stale = f"{m+6}ヶ月"
        bar = "#" * int(max(roi, 0) * 20) if roi == roi else ""
        print(f" {m:2} | {stale:>8} | {n:4} | {roi:.3f} | {hit:4.1f}  {bar}")

    # 前半(1-6=7-12ヶ月古) vs 後半(7-12=13-18ヶ月古)
    h1 = [p for m in range(1, 7) for p in by_month.get(m, [])]
    h2 = [p for m in range(7, 13) for p in by_month.get(m, [])]
    n1, r1, _ = _roi(h1)
    n2, r2, _ = _roi(h2)
    print(f"\n前半(1-6月, 7-12ヶ月古): n={n1} ROI={r1:.3f}")
    print(f"後半(7-12月, 13-18ヶ月古): n={n2} ROI={r2:.3f}")
    print(f"→ 前半>>後半 なら陳腐化あり=頻繁な再学習で改善余地。差が無ければ陳腐化は主因でない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
