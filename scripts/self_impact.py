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
        db_o1_rows = []
        for i in range(0, len(rids), 2000):
            ch = rids[i:i + 2000]
            db_o1_rows += db.query("""
                SELECT (o.year||o.month_day||o.jyo_cd||o.kaiji||o.nichiji||o.race_num) rid, o.umaban um,
                       CASE WHEN o.fuku_odds_low  ~ '^[0-9]+$' AND o.fuku_odds_low::numeric>0
                            THEN o.fuku_odds_low::numeric/10.0*100 END  lo,
                       CASE WHEN o.fuku_odds_high ~ '^[0-9]+$' AND o.fuku_odds_high::numeric>0
                            THEN o.fuku_odds_high::numeric/10.0*100 END hi,
                       CASE WHEN r.syusso_tosu ~ '^[0-9]+$' THEN r.syusso_tosu::int END fs
                FROM nl_o1 o JOIN nl_ra r USING (year,month_day,jyo_cd,kaiji,nichiji,race_num)
                WHERE (o.year||o.month_day||o.jyo_cd||o.kaiji||o.nichiji||o.race_num) = ANY(%(ids)s)""",
                {"ids": ch})
    finally:
        db.close()

    pool = defaultdict(int)
    for (rid, um), v in votes.items():
        pool[rid] += v

    # ★払戻式は当てにいかない。実払戻から「その馬の配当原資」を逆算する:
    #     原資_i = (実払戻 − 100)/100 × 自馬票数(円)
    #   自分が X 円入れると 自馬票数 += X、プール += X で、原資は控除率ぶんだけ目減りする:
    #     新払戻 = 100 + [原資_i − (1−r)·X/k] ÷ (自馬票数 + X) × 100
    #   支配的なのは分母の希釈で、これは票数と実払戻だけで決まる(式の形を知らなくてよい)。
    UNIT = 100.0   # 票数は100円単位(JV-Data 票数=枚数)。下の検証で妥当性を確認する

    def share_yen(rid, um):
        s_i = votes.get((rid, um), 0) * UNIT
        ac = actual.get((rid, um))
        if s_i <= 0 or ac is None:
            return None
        return (ac - 100.0) / 100.0 * s_i, s_i

    # 検証: 着内馬の原資合計 ≈ プール×(1−r) − 着内馬の票数合計 (単位・控除率・着内頭数の妥当性)
    rel = []
    for rid in {r for r, _ in votes}:
        pl = placed.get(rid) or set()
        if not pl:
            continue
        tot_share = 0.0; tot_s = 0.0; okall = True
        for u in pl:
            sv = share_yen(rid, u)
            if sv is None:
                okall = False; break
            tot_share += sv[0]; tot_s += sv[1]
        if not okall:
            continue
        P = pool[rid] * UNIT
        expect = P * (1 - TAKEOUT) - tot_s
        if expect > 0:
            rel.append(tot_share / expect)
    if rel:
        rel.sort()
        med = rel[len(rel) // 2]
        inb = sum(1 for x in rel if 0.95 < x < 1.05) / len(rel)
        print(f"\n[単位・控除率の検証] 着内馬の配当原資合計 ÷ (プール×{1-TAKEOUT:.1f} − 着内票数合計): "
              f"n={len(rel):,}  中央値 {med:.3f}  ±5%内 {inb:.1%}")
        if not (0.9 < med < 1.1):
            print(f"  ★1.0 から離れている。票数単位({int(UNIT)}円)か控除率({TAKEOUT})が違う可能性。"
                  "比が一定なら下の ROI の**傾き**は使えるが、水準は補正が要る")
    else:
        print("\n[単位・控除率の検証] 検証できる的中レースがありません")

    # ② 賭け金ごとの ROI
    print(f"\n[賭け金ごとの ROI]  基準は ¥100(自己インパクト無視)")
    n = len(sel)
    base_ret = sum(p for _r, _u, p in sel)
    miss = sum(1 for rid, um, pay in sel if pay > 0 and share_yen(rid, um) is None)
    print(f"  {'1点':>9}{'総投資':>12}{'ROI':>8}{'利益':>14}   的中時の平均払戻(100円あたり)")
    for x in (float(v) for v in args.stakes.split(",")):
        ret = 0.0; pays = []
        for rid, um, pay in sel:
            if pay <= 0:
                continue
            sv = share_yen(rid, um)
            if sv is None:                      # 票数が無い → 希釈無しの実払戻で代替
                use = float(pay)
            else:
                share, s_i = sv
                k = max(len(placed.get(rid) or {1}), 1)
                use = 100.0 + (share - (1 - TAKEOUT) * x / k) / (s_i + x) * 100.0
                use = max(use, 100.0)           # JRA は元本割れ無し(最低100円)
            pays.append(use)
            ret += x / 100.0 * use
        roi = ret / (n * x)
        print(f"  {int(x):>9,}{int(n*x):>12,}{roi:>8.4f}{int(ret - n*x):>14,}   "
              f"{sum(pays)/max(len(pays),1):>8.1f}")
    if miss:
        print(f"  ※ 票数が引けず希釈を計算できなかった的中: {miss:,} 件(実払戻で代替=自己インパクト過小)")
    print(f"\n  ¥100 時の実績 ROI(参考, 実払戻ベース)= {base_ret/(n*100):.4f}")
    print("  ※ 単勝プールから取った信号で複勝を買うので、自己投票は信号源を汚さない(払戻だけが下がる)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
