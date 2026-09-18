"""自己インパクト: 1点いくらまで賭けると ROI が 1.0 を割るかを票数から厳密に出す。

flow_tan の選別は ROI 1.11〜1.17 だが、月170本・¥100 では8ヶ月で約¥23,000にしかならない。
規模を上げると**自分の投票が払戻を下げる**。オッズからは票数を逆算できないので、H1(票数)が要る。

複勝の払戻(100円あたり):
    100 + [(複勝プール×(1−控除率) − 的中馬の票数合計×100) ÷ 着内頭数] ÷ 自馬票数 × 100
★この式自体を nl_hr の実払戻と突き合わせて検証してから使う(推測した式で資金上限は決められない)。

自分が X 円入れると: プール += X, 自馬票数 += X/100, (的中時のみ)的中票数合計 += X。

    python scripts/meta_model_ts.py --cand ~/wf/2025/cand.csv ~/wf/2026/cand.csv \\
        --rolling --min-fit-months 4 --top-frac 0.05 --dump-selections ~/sel.csv
    python scripts/self_impact.py --selections ~/sel.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB

TAKEOUT = 0.20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", required=True)
    ap.add_argument("--stakes", default="100,1000,5000,10000,30000,50000,100000,200000,500000")
    args = ap.parse_args()

    sel = []
    with open(args.selections, encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            sel.append((row[0], row[1], int(row[2])))
    rids = sorted({s[0] for s in sel})
    print(f"採用馬券 {len(sel):,} 件 / {len(rids):,} レース")

    db = FeatureDB(load_features_config())
    try:
        votes, placed, paykey, actual = {}, defaultdict(set), {}, {}
        for i in range(0, len(rids), 2000):
            ch = rids[i:i + 2000]
            for r in db.query("""
                SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) rid, umaban,
                       fuku_vote, fuku_pay_key
                FROM nl_h1 WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num) = ANY(%(ids)s)
                  AND fuku_vote IS NOT NULL AND fuku_vote > 0""", {"ids": ch}):
                votes[(r["rid"], r["umaban"])] = int(r["fuku_vote"])
                paykey[r["rid"]] = r["fuku_pay_key"]
            for r in db.query("""
                SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) rid,
                       regexp_replace(kumi,'[^0-9]','','g') um, pay
                FROM nl_hr WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num) = ANY(%(ids)s)
                  AND bet_type='fuku'""", {"ids": ch}):
                um = r["um"].zfill(2)
                placed[r["rid"]].add(um)
                p = (r["pay"] or "").strip()
                if p.isdigit():
                    actual[(r["rid"], um)] = int(p)
    finally:
        db.close()

    pool = defaultdict(int)
    for (rid, um), v in votes.items():
        pool[rid] += v

    def payout(rid, um, extra_yen=0.0):
        """予測払戻(100円あたり)。extra_yen は自分の投入額。"""
        s_i = votes.get((rid, um), 0) * 100.0 + extra_yen
        if s_i <= 0 or rid not in pool:
            return None
        P = pool[rid] * 100.0 + extra_yen
        pl = placed.get(rid) or set()
        if not pl or um not in pl:
            return None
        k = len(pl)
        s_placed = sum(votes.get((rid, u), 0) * 100.0 for u in pl) + extra_yen
        prof = P * (1 - TAKEOUT) - s_placed
        if prof <= 0:
            return 100.0
        return 100.0 + (prof / k) / s_i * 100.0

    # ① 式の検証: 実払戻(nl_hr)と一致するか
    ok = tot = 0; err = []
    for rid, um, _pay in sel:
        pr = payout(rid, um)
        ac = actual.get((rid, um))
        if pr is None or ac is None:
            continue
        tot += 1
        e = abs(pr - ac) / max(ac, 1)
        ok += e < 0.03
        err.append(e)
    if not tot:
        print("★検証できる的中馬券がありません(nl_h1/nl_hr の突合に失敗)")
        return 1
    err.sort()
    print(f"\n[式の検証] 的中 {tot:,} 件で予測払戻 vs 実払戻(nl_hr): "
          f"3%以内一致 {ok:,}/{tot:,} = {ok/tot:.1%}  中央誤差 {err[len(err)//2]:.2%}")
    if ok / tot < 0.9:
        print("★式が合っていません。控除率/着内頭数/票数の単位を見直すこと。以下は参考値です。")

    # ② 賭け金ごとの ROI
    print(f"\n[賭け金ごとの ROI]  基準は ¥100(自己インパクト無視)")
    base_ret = sum(p for _r, _u, p in sel)
    n = len(sel)
    print(f"  {'1点':>9}{'総投資':>12}{'ROI':>8}{'利益':>12}   的中時の平均払戻(100円あたり)")
    for x in (float(v) for v in args.stakes.split(",")):
        ret = 0.0; pays = []
        for rid, um, pay in sel:
            if pay <= 0:
                continue
            pr = payout(rid, um, extra_yen=x)
            use = pr if pr is not None else float(pay)
            pays.append(use)
            ret += x / 100.0 * use
        roi = ret / (n * x)
        print(f"  {int(x):>9,}{int(n*x):>12,}{roi:>8.4f}{int(ret - n*x):>12,}   "
              f"{sum(pays)/max(len(pays),1):>8.1f}")
    print(f"\n  ¥100 時の実績 ROI(参考, 実払戻ベース)= {base_ret/(n*100):.4f}")
    print("  ※ 単勝プールから取った信号で複勝を買うので、自己投票は信号源を汚さない(払戻だけが下がる)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
