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

    # ① 票数ブロックと払戻式を**同時に**検証する。
    #    nl_o1 に複勝オッズの下限/上限があるのは、払戻が「他にどの馬が来るか」に依存するから。
    #    = 利益を着内頭数で割る形でなければ幅が出ない(プールを単純に割る式なら定数になる)。
    #    下限 = 他の着内馬が最も売れている2頭のとき、上限 = 最も売れていない2頭のとき。
    #    これが nl_o1 と一致すれば、[503] が複勝票数であることと式の形が同時に確定する。
    def bounds(rid, um, k):
        s_i = votes.get((rid, um), 0) * 100.0
        if s_i <= 0:
            return None
        others = sorted(v * 100.0 for (r2, u2), v in votes.items() if r2 == rid and u2 != um)
        if len(others) < k - 1:
            return None
        P = pool[rid] * 100.0
        hi_others = sum(others[-(k - 1):])   # 最も売れている → 払戻は下限
        lo_others = sum(others[:k - 1])      # 最も売れていない → 払戻は上限
        def f(o):
            prof = P * (1 - TAKEOUT) - (s_i + o)
            return 100.0 + (prof / k) / s_i * 100.0 if prof > 0 else 100.0
        return f(hi_others), f(lo_others)

    okl = okh = totb = 0; errl = []
    for r in db_o1_rows:
        rid, um = r["rid"], r["um"]
        k = 3 if (r["fs"] or 0) >= 8 else 2
        bd = bounds(rid, um, k)
        if bd is None or not r["lo"] or not r["hi"]:
            continue
        totb += 1
        pl, ph = bd
        okl += abs(pl - r["lo"]) / r["lo"] < 0.03
        okh += abs(ph - r["hi"]) / r["hi"] < 0.03
        errl.append(abs(pl - r["lo"]) / r["lo"])
    if totb:
        errl.sort()
        print(f"\n[票数ブロック+式の検証] nl_o1 の複勝オッズ下限/上限を票数から予測: n={totb:,}  "
              f"下限一致 {okl/totb:.1%}  上限一致 {okh/totb:.1%}  下限の中央誤差 {errl[len(errl)//2]:.2%}")
        if okl / totb < 0.9:
            print("  ★一致しない。[503] が複勝票数でないか、控除率/着内頭数が違う。"
                  "probe で複勝ブロックを総当たり検証すること")
    else:
        print("\n[票数ブロック+式の検証] 検証データが取れませんでした")

    # ② 式の検証: 実払戻(nl_hr)と一致するか
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
