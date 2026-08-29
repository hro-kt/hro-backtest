"""三連複候補の ROI を「レース特徴」でスライスし、確定オッズでも勝てる部分市場を探す。

sweep --save-candidates の CSV(bet_type,er,prob,odds,settled,hit,payout,seg_runs,seg_layoff,
race_id,selection_id) を読み、race_id で feat_matrix のレース特徴(頭数/グレード/馬場/芝ダ/距離)を
引いてセグメント別に ROI(=Σpayout/Σstake, flat¥100)・的中率・n を出す。

★規律: 全体で負けても特定セグメントで勝てる所を探すのが目的。ただし多重検定で偽陽性が出るので、
ここで見つけた候補は必ず別OOS窓でも再確認すること。ROI>1.0 かつ n が十分(既定>=300)のみ着目。

使い方(hro-backtest, 学習時と同じ HRO_ABLATE_* env):
    python scripts/trio_segment_slice.py /tmp/cand2025_oos.csv --er 1.7 --prob 0.0 --min-n 300
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from hro_features.config import load_config
from hro_features.db import FeatureDB


def _dist_band(d):
    try:
        d = int(d)
    except (TypeError, ValueError):
        return "?"
    if d <= 1400:
        return "1_sprint(<=1400)"
    if d <= 1800:
        return "2_mile(1401-1800)"
    if d <= 2200:
        return "3_mid(1801-2200)"
    return "4_long(>2200)"


def _field_band(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n <= 10:
        return "a_small(<=10)"
    if n <= 13:
        return "b_mid(11-13)"
    if n <= 16:
        return "c_large(14-16)"
    return "d_full(17-18)"


def _odds_band(o):
    o = float(o)
    for hi, lab in [(6, "1_(1.5-6)"), (12, "2_(6-12)"), (25, "3_(12-25)"),
                    (50, "4_(25-50)"), (100, "5_(50-100)"), (300, "6_(100-300)")]:
        if o < hi:
            return lab
    return "7_(300+)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="sweep --save-candidates の CSV")
    ap.add_argument("--er", type=float, default=1.7, help="min_er")
    ap.add_argument("--prob", type=float, default=0.0, help="min_prob")
    ap.add_argument("--min-n", type=int, default=300, help="着目する最小n")
    args = ap.parse_args()

    # 1) 候補を読み、閾値通過 & trio & settled のみ採用
    bets = []           # (race_id, odds, hit, payout)
    race_ids = set()
    with open(args.csv, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            bt, er, prob, odds = row[0], float(row[1]), float(row[2]), float(row[3])
            settled, hit, payout = row[4] == "True", row[5] == "True", int(row[6])
            rid, sel = (row[9] if len(row) > 9 else ""), (row[10] if len(row) > 10 else "")
            if bt != "trio" or not settled:
                continue
            if er < args.er or prob < args.prob:
                continue
            bets.append((rid, odds, hit, payout))
            race_ids.add(rid)
    if not bets:
        print("該当候補なし(閾値/CSVを確認)")
        return 1

    # 2) レース特徴を feat_matrix から一括取得(レース定数=maxで代表)
    db = FeatureDB(load_config())
    feats = {}
    ids = list(race_ids)
    RID = "(year||month_day||jyo_cd||kaiji||nichiji||race_num)"
    for i in range(0, len(ids), 5000):
        chunk = ids[i:i + 5000]
        rows = db.query(
            f"SELECT {RID} rid, max(field_size) fs, max(grade_cd) grade, "
            f"max(surface) surf, max(baba_state) baba, max(distance_m) dist "
            f"FROM feat_matrix WHERE {RID} = ANY(%(ids)s) GROUP BY 1", {"ids": chunk})
        for x in rows:
            feats[x["rid"]] = x
    db.close()

    # 3) セグメント別に集計
    def agg():
        return [0, 0, 0]  # [n, stake, payout]

    dims = {"頭数": _field_band, "距離": _dist_band, "オッズ帯": None,
            "芝ダ": None, "馬場": None, "グレード": None}
    seg = {d: defaultdict(agg) for d in dims}
    overall = agg()

    for rid, odds, hit, payout in bets:
        fr = feats.get(rid)
        keys = {
            "頭数": _field_band(fr["fs"]) if fr else "?",
            "距離": _dist_band(fr["dist"]) if fr else "?",
            "オッズ帯": _odds_band(odds),
            "芝ダ": (str(fr["surf"]) if fr and fr["surf"] is not None else "?"),
            "馬場": (str(fr["baba"]) if fr and fr["baba"] is not None else "?"),
            "グレード": (str(fr["grade"]) if fr and fr["grade"] is not None else "?"),
        }
        overall[0] += 1; overall[1] += 100; overall[2] += payout
        for d, k in keys.items():
            s = seg[d][k]
            s[0] += 1; s[1] += 100; s[2] += payout

    n, stake, pay = overall
    print(f"\n== 全体 == n={n:,} ROI={pay/stake:.3f} hitはpayout>0で近似")
    print(f"  (閾値 er>={args.er} prob>={args.prob}, min-n={args.min_n})")
    for d in dims:
        print(f"\n== {d} 別 ==")
        rows = sorted(seg[d].items(), key=lambda kv: (kv[1][2] / kv[1][1]) if kv[1][1] else 0, reverse=True)
        for k, (nn, st, py) in rows:
            roi = py / st if st else 0
            flag = "  ★" if (roi > 1.0 and nn >= args.min_n) else ""
            print(f"  {k:20} n={nn:>7,} ROI={roi:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
