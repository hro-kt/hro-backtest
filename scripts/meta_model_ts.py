"""本番形の meta-model: 決定時点(T−lead秒)のスナップショットオッズ + 直前フロー + 凍結 fundamental。

Benter 型(市場 baseline + 凍結 p_fund を少数パラメータで統合し EV で選ぶ)は、確定オッズ q で
ローリング6窓中5窓が正(平均 +3.1pt)。しかし本番で T−30s に見えるのは確定オッズではない。
ここでは q を ts_o1 の T−lead 時点の値に置き換え、直前フロー(T−flow分 → T−lead の
レース内シェアの logit 差)を項として足し、EV = p* × q_snap で選んで、払戻は実際(nl_hr)で決済する。
= そのまま運用に載せられる形のバックテスト(確定オッズの楽観上限ではない)。

  logit(p*) = a + b·logit(q_snap) + c·logit(p_fund) + d·flow

p_fund は候補CSV(sweep --save-candidates, place)の prob(市場を見せずに作ったモデルの PL 複勝確率)。
fit/eval は日付で時系列分割。同一本数 top-n で「生 p_fund 確率順」と比較(対応のあるブートストラップ)。

    python scripts/meta_model_ts.py --cand ~/wf/2025/cand.csv ~/wf/2026/cand.csv \\
        --fit-to 20260430 --eval-from 20260501 --lead-sec 30 --flow-min 5 --top-frac 0.10
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB

SQL = """
WITH ra AS (
  SELECT year, month_day, jyo_cd, kaiji, nichiji, race_num,
         to_timestamp(year||month_day||hasso_time, 'YYYYMMDDHH24MI') AS post_ts
  FROM nl_ra
  WHERE jyo_cd BETWEEN '01' AND '10' AND year||month_day BETWEEN %(d0)s AND %(d1)s
    AND hasso_time ~ '^[0-9]{4}$'
),
s1 AS (  -- 決定時点: 発走 −lead 秒 以前の最新
  SELECT DISTINCT ON (t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban)
         t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban, t.fuku_odds_low AS f1,
         t.tan_odds AS t1,
         EXTRACT(EPOCH FROM (ra.post_ts - to_timestamp(t.year||t.hasso_time,'YYYYMMDDHH24MI'))) AS lead1
  FROM ts_o1 t JOIN ra USING (year,month_day,jyo_cd,kaiji,nichiji,race_num)
  WHERE to_timestamp(t.year||t.hasso_time,'YYYYMMDDHH24MI') <= ra.post_ts - make_interval(secs => %(lead)s)
    AND t.fuku_odds_low ~ '^[0-9]+$' AND t.fuku_odds_low::numeric > 0
  ORDER BY t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban, t.hasso_time DESC
),
s0 AS (  -- フローの起点: 発走 −flow 分 以前の最新
  SELECT DISTINCT ON (t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban)
         t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban, t.fuku_odds_low AS f0,
         t.tan_odds AS t0,
         EXTRACT(EPOCH FROM (ra.post_ts - to_timestamp(t.year||t.hasso_time,'YYYYMMDDHH24MI'))) AS lead0
  FROM ts_o1 t JOIN ra USING (year,month_day,jyo_cd,kaiji,nichiji,race_num)
  WHERE to_timestamp(t.year||t.hasso_time,'YYYYMMDDHH24MI') <= ra.post_ts - make_interval(mins => %(flow)s)
    AND t.fuku_odds_low ~ '^[0-9]+$' AND t.fuku_odds_low::numeric > 0
  ORDER BY t.year,t.month_day,t.jyo_cd,t.kaiji,t.nichiji,t.race_num,t.umaban, t.hasso_time DESC
)
SELECT s1.year||s1.month_day||s1.jyo_cd||s1.kaiji||s1.nichiji||s1.race_num AS rid, s1.umaban,
       s1.f1::numeric/10.0 AS q1, s0.f0::numeric/10.0 AS q0,
       s1.lead1, s0.lead0,
       CASE WHEN s1.t1 ~ '^[0-9]+$' AND s1.t1::numeric>0 THEN s1.t1::numeric/10.0 END AS tan1,
       CASE WHEN s0.t0 ~ '^[0-9]+$' AND s0.t0::numeric>0 THEN s0.t0::numeric/10.0 END AS tan0,
       (SELECT h.pay FROM nl_hr h
         WHERE (h.year,h.month_day,h.jyo_cd,h.kaiji,h.nichiji,h.race_num)
             = (s1.year,s1.month_day,s1.jyo_cd,s1.kaiji,s1.nichiji,s1.race_num)
           AND h.bet_type=%(hr)s AND regexp_replace(h.kumi,'[^0-9]','','g') = s1.umaban LIMIT 1) AS pay
FROM s1 JOIN s0 USING (year,month_day,jyo_cd,kaiji,nichiji,race_num,umaban)
"""


def lg(x):
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def load_cand(paths):
    pf = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if row[0] != "place" or row[4] != "True":
                    continue
                rid = row[9] if len(row) > 9 else ""
                sel = row[10] if len(row) > 10 else ""
                if rid and sel.strip().isdigit():
                    pf[(rid, f"{int(sel):02d}")] = float(row[2])
    return pf


def irls(X, y, l2=1e-6):
    w = np.zeros(X.shape[1])
    for _ in range(60):
        mu = 1 / (1 + np.exp(-(X @ w))); W = mu * (1 - mu) + 1e-9
        g = X.T @ (y - mu); H = (X * W[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def level(agg, iters=10000, seed=7):
    """プールした ROI 自体の CI と P(ROI<=1.0)。基準との差だけでは「勝てるか」を答えられない。"""
    rids = sorted(agg)
    st = np.array([agg[r][0] for r in rids]); pa = np.array([agg[r][1] for r in rids])
    rng = np.random.default_rng(seed); k = len(rids); out = np.empty(iters); done = 0
    while done < iters:
        m = min(200, iters - done); idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = pa[idx].sum(1) / st[idx].sum(1); done += m
    d = np.sort(out)
    return (pa.sum() / st.sum(), float(d[int(.025 * iters)]), float(d[int(.975 * iters)]),
            float((d <= 1.0).mean()))


def paired(aggA, aggB, iters=10000, seed=42):
    rids = sorted(set(aggA) | set(aggB))
    sa = np.array([aggA.get(r, (0, 0))[0] for r in rids]); pa = np.array([aggA.get(r, (0, 0))[1] for r in rids])
    sb = np.array([aggB.get(r, (0, 0))[0] for r in rids]); pb = np.array([aggB.get(r, (0, 0))[1] for r in rids])
    ra, rb = pa.sum() / sa.sum(), pb.sum() / sb.sum()
    rng = np.random.default_rng(seed); k = len(rids); out = np.empty(iters); done = 0
    while done < iters:
        m = min(200, iters - done); idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = pb[idx].sum(1) / sb[idx].sum(1) - pa[idx].sum(1) / sa[idx].sum(1); done += m
    d = np.sort(out)
    return ra, rb, rb - ra, float(d[int(.025 * iters)]), float(d[int(.975 * iters)]), float((d <= 0).mean())


def pick(items, score, thr):
    """score が絶対閾値 thr 以上のものを全部買う(¥100)。運用でそのまま実行できる形。

    ★top-frac(eval期間全体の上位◯%)は**未来のスコア分布を知らないと閾値が決まらない**ので
      実運用では使えない定義だった。fit 期間で決めた絶対値を eval に適用する。
      1レース固定N点だと信号の弱いレースでも無理に買って薄まる(実測: 複勝 1.17→1.006)。
      効果の大半は「どのレースで賭けるか」の選別が担っている。
    """
    return [it for it in items if score(it) >= thr]


def agg_from(sel):
    agg = defaultdict(lambda: [0.0, 0.0])
    for rid, pay, _f in sel:
        agg[rid][0] += 100; agg[rid][1] += pay
    return {k: tuple(v) for k, v in agg.items()}


def agg_topn(items, score, n, per_race=0):
    """items: list of (rid, pay, feats)。score の大きい順に買う(¥100)。

    per_race>0 なら **1レースあたり上位 per_race 点**(全体 top-n ではなく)。
    """
    if per_race > 0:
        by_r = defaultdict(list)
        for it in items:
            by_r[it[0]].append(it)
        ranked = []
        for _rid, v in by_r.items():
            ranked += sorted(v, key=lambda it: -score(it))[:per_race]
    else:
        ranked = sorted(items, key=lambda it: -score(it))[:n]
    agg = defaultdict(lambda: [0.0, 0.0])
    for rid, pay, _ in ranked:
        agg[rid][0] += 100; agg[rid][1] += pay
    return {k: tuple(v) for k, v in agg.items()}


COMBO_SQL = """
SELECT (o.year||o.month_day||o.jyo_cd||o.kaiji||o.nichiji||o.race_num) rid,
       o.kumi, {odds_expr} AS odds,
       (SELECT h.pay FROM nl_hr h
         WHERE (h.year,h.month_day,h.jyo_cd,h.kaiji,h.nichiji,h.race_num)
             = (o.year,o.month_day,o.jyo_cd,o.kaiji,o.nichiji,o.race_num)
           AND h.bet_type=%(hr)s AND regexp_replace(h.kumi,'[^0-9]','','g') = o.kumi LIMIT 1) AS pay
FROM {tbl} o
WHERE (o.year||o.month_day||o.jyo_cd||o.kaiji||o.nichiji||o.race_num) = ANY(%(ids)s)
  AND {odds_raw} ~ '^[0-9]+$' AND {odds_raw}::numeric > 0
"""


def load_combos(db, rids, bet_type):
    """組の確定オッズと払戻。kumi は '0102'(ワイド) / '010203'(三連複) の数字連結。"""
    if bet_type == "wide":
        tbl, raw, hr, legs = "nl_o3", "o.odds_low", "wide", 2
    else:
        tbl, raw, hr, legs = "nl_o5", "o.odds", "sanrenfuku", 3
    sql = COMBO_SQL.format(tbl=tbl, odds_raw=raw, odds_expr=f"{raw}::numeric/10.0")
    out = []
    for i in range(0, len(rids), 800):
        for r in db.query(sql, {"ids": rids[i:i + 800], "hr": hr}):
            k = r["kumi"]
            if len(k) != legs * 2 or not k.isdigit():
                continue
            ums = tuple(f"{int(k[j*2:j*2+2]):02d}" for j in range(legs))
            pay = int(str(r["pay"]).strip()) if r["pay"] and str(r["pay"]).strip().isdigit() else 0
            out.append((r["rid"], ums, float(r["odds"]), pay))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True, help="place 候補CSV(p_fund の出どころ)")
    ap.add_argument("--from", dest="d0", default="20250901"); ap.add_argument("--to", dest="d1", default="20260831")
    ap.add_argument("--fit-to", default=None, help="YYYYMMDD。これ以前で fit(--rolling 時は不要)")
    ap.add_argument("--eval-from", default=None, help="YYYYMMDD。これ以降で eval(--rolling 時は不要)")
    ap.add_argument("--lead-sec", type=int, default=30, help="決定時点 = 発走 −これ秒")
    ap.add_argument("--flow-min", type=int, default=5, help="フローの起点 = 発走 −これ分")
    ap.add_argument("--bet-type", choices=("place", "win", "wide", "trio"), default="place",
                    help="賭ける券種。★信号(flow_tan)は単勝プールから取るので、複勝で賭ければ信号源と"
                         "別プール=自己投票が信号を汚さない。単勝で賭けると同じプールを自分で動かす。"
                         "両方で効くなら信号が本物である強い傍証(控除率はどちらも20%)。"
                         "wide/trio は組の券種: 信号は単勝プール(ts_o1)から作り、"
                         "組のスコアは構成馬の flow_tan の**最小値**(全頭が買われている組だけを採る)。"
                         "★決定時点のワイド/三連複オッズは時系列が無い(ts_sokuho_o3/o5 は5日分)ので"
                         "確定オッズで EV を計算する＝**楽観側の上限測定**。効いてから運用形を考える")
    ap.add_argument("--top-frac", type=float, default=0.10, help="eval 内で買う割合(同一本数比較)")
    ap.add_argument("--thr-quantile", type=float, default=None,
                    help="fit 期間のスコア分布のこの上側分位(例 0.99)を**絶対閾値**にして eval に適用。"
                         "未来を見ないので運用でそのまま実行できる。top-frac/per-race より優先")
    ap.add_argument("--per-race", type=int, default=0,
                    help="1レースあたり上位N点を買う(>0 で --top-frac より優先)。"
                         "運用ルールに直結し、特定レースへの集中か広く薄くかも区別できる")
    ap.add_argument("--rolling", action="store_true",
                    help="月単位の拡大窓ローリング。各 eval 月について『その月より前の全データ』で fit し、"
                         "月ごとに top-frac を買って全月をプール。eval を 4ヶ月→8ヶ月に増やし月別の再現も見る"
                         "(ts_o1 は1年しか無く、単一分割の eval 1,371本では CI が ±0.2 で判定不能だった)")
    ap.add_argument("--min-fit-months", type=int, default=4, help="ローリングの最小 fit 月数")
    ap.add_argument("--dump-selections", default=None,
                    help="採用した馬券を CSV 出力(自己インパクト計算 self_impact.py の入力)")
    ap.add_argument("--dump-variant", default="flow_tan 単独(単勝プール)", help="出力する変種名")
    args = ap.parse_args()
    if not args.rolling and not (args.fit_to and args.eval_from):
        ap.error("--rolling を使わない場合は --fit-to と --eval-from が要ります")

    pf = load_cand(args.cand)
    db = FeatureDB(load_features_config())
    try:
        rows = db.query(SQL, {"d0": args.d0, "d1": args.d1, "lead": args.lead_sec,
                              "flow": args.flow_min,
                              "hr": "fuku" if args.bet_type == "place" else "tan"})
    finally:
        db.close()
    # レース内シェアで flow を作る
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["rid"]].append(r)
    # ★リーク監査: 採用したスナップショットが本当に締切前か。ts_o1 の発表時刻は発走をまたぐ
    #   ものもあるので、post_ts − snap_ts が 0 以下なら締切後の値を拾っている＝結果リーク。
    l1 = sorted(float(r["lead1"]) for r in rows); l0 = sorted(float(r["lead0"]) for r in rows)
    def q(a, f): return a[min(len(a) - 1, int(len(a) * f))]
    bad = sum(1 for v in l1 if v <= 0)
    print(f"リード時間監査(post − snap 秒): 決定時点 中央値 {q(l1,.5):,.0f}s "
          f"[p05 {q(l1,.05):,.0f} / p95 {q(l1,.95):,.0f}]   起点 中央値 {q(l0,.5):,.0f}s")
    print(f"  締切後(<=0s)を拾った行: {bad:,} / {len(rows):,}"
          + ("  ★リーク。SQL の条件を見直すこと" if bad else "  (0件=締切前のみ)"))

    items = []   # (rid, pay, dict)
    miss = 0
    for rid, rs in by_race.items():
        s1 = sum(1 / float(r["q1"]) for r in rs); s0 = sum(1 / float(r["q0"]) for r in rs)
        # 単勝プール: 1/tan_odds はプール占有率そのもの(複勝下限は他馬の組合せに依存し粗い)
        ts1 = sum(1 / float(r["tan1"]) for r in rs if r["tan1"])
        ts0 = sum(1 / float(r["tan0"]) for r in rs if r["tan0"])
        for r in rs:
            p_fund = pf.get((rid, f"{int(r['umaban']):02d}"))
            if p_fund is None:
                miss += 1; continue
            q1 = float(r["q1"]); share1 = (1 / q1) / s1; share0 = (1 / float(r["q0"])) / s0
            # EV に使うオッズは賭ける券種のもの。単勝なら tan1(決定時点の単勝オッズ)
            q_bet = q1 if args.bet_type == "place" else (float(r["tan1"]) if r["tan1"] else None)
            if q_bet is None:
                miss += 1; continue
            flow_p = lg(share1) - lg(share0)
            if r["tan1"] and r["tan0"] and ts1 > 0 and ts0 > 0:
                flow_t = lg((1 / float(r["tan1"])) / ts1) - lg((1 / float(r["tan0"])) / ts0)
            else:
                flow_t = 0.0
            pay = int(str(r["pay"]).strip()) if r["pay"] is not None and str(r["pay"]).strip().isdigit() else 0
            # ★決定時点(T−60s)での単勝/複勝プール乖離の**水準**。11年検証で現象の主成分は
            #   「変化」ではなく「確定時点の単勝占有率の水準」(level_end +0.109 vs level_start +0.014)
            #   = Hausch–Ziemba の place/show 非効率。確定オッズは決定時点に見えないが、
            #   同じ乖離を T−60s で測れば使える(単勝・複勝とも見えている)。1時点で済むので運用が単純。
            xlv = (lg((1 / float(r["tan1"])) / ts1) - lg(share1)) if (r["tan1"] and ts1 > 0) else 0.0
            items.append((rid, pay, {"ymd": rid[:8], "um": f"{int(r['umaban']):02d}",
                                     "q1": q_bet, "qimp": min(0.8 / q_bet, 0.98),
                                     "flow": flow_p, "flow_tan": flow_t,
                                     "xpool": flow_t - flow_p,   # 変化の差(単勝が先行し複勝が未反応)
                                     "xpool_level": xlv,         # 水準の差(決定時点の乖離そのもの)
                                     "pf": p_fund}))
    print(f"ts_o1 結合 {len(rows):,} 行 → 有効 {len(items):,} (欠落 {miss:,})   "
          f"券種={args.bet_type}, 決定時点 T−{args.lead_sec}s, フロー起点 T−{args.flow_min}m")

    if args.bet_type in ("wide", "trio"):
        # 馬ごとの flow_tan を引けるようにして、組に展開し直す。
        # 組のスコア = 構成馬の flow_tan の**最小値**(一部の脚だけ買われている組を除く)。
        # EV のオッズは確定オッズ(決定時点の組オッズは時系列が無い)＝楽観側の上限測定。
        per_horse = {(rid, f["um"]): f for rid, _p, f in items}
        db2 = FeatureDB(load_features_config())
        try:
            combos = load_combos(db2, sorted({rid for rid, _p, _f in items}), args.bet_type)
        finally:
            db2.close()
        new_items, dropped = [], 0
        for rid, ums, odds, pay in combos:
            fs = [per_horse.get((rid, u)) for u in ums]
            if any(f is None for f in fs):
                dropped += 1; continue
            new_items.append((rid, pay, {
                "ymd": rid[:8], "um": "-".join(ums), "q1": odds,
                "qimp": min((1 - 0.225 if args.bet_type == "wide" else 1 - 0.25) / odds, 0.98),
                "flow": min(f["flow"] for f in fs),
                "flow_tan": min(f["flow_tan"] for f in fs),
                "xpool": min(f["xpool"] for f in fs),
                "xpool_level": min(f["xpool_level"] for f in fs),
                "pf": max(min(f["pf"] for f in fs), 1e-6),
            }))
        items = new_items
        print(f"  → {args.bet_type} の組に展開: {len(items):,} 点 "
              f"(脚の flow が引けず除外 {dropped:,})  組スコア=構成馬の最小 flow_tan")
    def design(its, with_flow):
        cols = [np.ones(len(its)),
                np.array([lg(i[2]["qimp"]) for i in its]),
                np.array([lg(i[2]["pf"]) for i in its])]
        if with_flow:
            cols.append(np.array([i[2]["flow"] for i in its]))
        return np.column_stack(cols)
    def pstar(w, it, with_flow):
        z = w[0] + w[1] * lg(it[2]["qimp"]) + w[2] * lg(it[2]["pf"]) + (w[3] * it[2]["flow"] if with_flow else 0)
        return 1 / (1 + math.exp(-z))

    def fit_weights(fit_items):
        y = np.array([1.0 if i[1] > 0 else 0.0 for i in fit_items])
        return irls(design(fit_items, False), y), irls(design(fit_items, True), y)

    def variants_for(wB, wF):
        return [
            ("生p_fund EV順(q_snap)", lambda it: it[2]["pf"] * it[2]["q1"]),
            ("Benter(snap) EV順",     lambda it: pstar(wB, it, False) * it[2]["q1"]),
            ("Benter+flow EV順",      lambda it: pstar(wF, it, True) * it[2]["q1"]),
            ("flow 単独(順位)",        lambda it: it[2]["flow"]),
            ("flow_tan 単独(単勝プール)", lambda it: it[2]["flow_tan"]),
            ("xpool(単勝先行-複勝未反応)", lambda it: it[2]["xpool"]),
            ("xpool_level(水準乖離)",   lambda it: it[2]["xpool_level"]),
            ("xpool_level+flow_tan",   lambda it: it[2]["xpool_level"] + it[2]["flow_tan"]),
        ]
    NAMES = [v[0] for v in variants_for(np.zeros(3), np.zeros(4))]

    def report(base_agg, agg_by_name, sel_by_name, label):
        print(f"\n[{label}]  EV は q_snap(決定時点のオッズ)で計算、払戻は実績")
        rb0 = sum(v[1] for v in base_agg.values()) / sum(v[0] for v in base_agg.values())
        nb = int(sum(v[0] for v in base_agg.values()) // 100)
        print(f"  {'基準 生p_fund確率順':<22} ROI {rb0:.4f}  n={nb:,}")
        for nm in NAMES:
            ra, rb, d, lo, hi, pp = paired(base_agg, agg_by_name[nm])
            sel = sel_by_name[nm]
            hit = sum(1 for _r, pay, _f in sel if pay > 0) / max(len(sel), 1)
            mo = sum(f["q1"] for _r, _p, f in sel) / max(len(sel), 1)
            lv, llo, lhi, p1 = level(agg_by_name[nm])
            print(f"  {nm:<22} ROI {rb:.4f} [{llo:.3f},{lhi:.3f}] P(ROI<=1)={p1:.3f}"
                  f"  差={d:+.4f} [{lo:+.4f},{hi:+.4f}] P(差<=0)={pp:.3f}"
                  f"  的中={hit:.1%} odds={mo:.2f}")

    if not args.rolling:
        fit = [it for it in items if it[2]["ymd"] <= args.fit_to]
        ev = [it for it in items if it[2]["ymd"] >= args.eval_from]
        print(f"fit {len(fit):,} 本 / eval {len(ev):,} 本 ({len({i[0] for i in ev}):,} レース)")
        wB, wF = fit_weights(fit)
        print(f"  Benter(snap) : logit p* = {wB[0]:+.3f} + {wB[1]:.3f}·logit(q_snap) + {wB[2]:.3f}·logit(p_fund)")
        print(f"  +flow        : logit p* = {wF[0]:+.3f} + {wF[1]:.3f}·logit(q_snap) + {wF[2]:.3f}·logit(p_fund) + {wF[3]:+.3f}·flow")
        n = max(1, int(len(ev) * args.top_frac))
        base = agg_topn(ev, lambda it: it[2]["pf"], n, args.per_race)
        agg_by, sel_by = {}, {}
        for nm, sc in variants_for(wB, wF):
            agg_by[nm] = agg_topn(ev, sc, n, args.per_race)
            sel_by[nm] = sorted(ev, key=lambda it: -sc(it))[:n]
        report(base, agg_by, sel_by, f"eval 同一本数 top-{n:,}  基準=生 p_fund 確率順")
        return 0

    # ---- 拡大窓ローリング(月単位) ----
    months = sorted({it[2]["ymd"][:6] for it in items})
    mode = (f"fit分位 {args.thr_quantile} の絶対閾値" if args.thr_quantile is not None
            else (f"1R上位{args.per_race}点" if args.per_race else f"eval上位{args.top_frac:.0%}"))
    print(f"ローリング: 月 {months[0]}〜{months[-1]} ({len(months)}ヶ月), 最小fit {args.min_fit_months}ヶ月, "
          f"選別={mode}")
    base_all = defaultdict(lambda: [0.0, 0.0])
    agg_all = {nm: defaultdict(lambda: [0.0, 0.0]) for nm in NAMES}
    sel_all = {nm: [] for nm in NAMES}
    per_month = []
    for i, m in enumerate(months):
        if i < args.min_fit_months:
            continue
        fit = [it for it in items if it[2]["ymd"][:6] < m]
        ev = [it for it in items if it[2]["ymd"][:6] == m]
        if len(ev) < 200:
            continue
        wB, wF = fit_weights(fit)
        n = max(1, int(len(ev) * args.top_frac))
        use_thr = args.thr_quantile is not None
        if use_thr:
            # fit 期間の分位から各変種の絶対閾値を決める(eval のスコアは一切見ない)
            thr = {}
            for nm, sc in variants_for(wB, wF):
                vals = sorted(sc(it) for it in fit)
                thr[nm] = vals[min(len(vals) - 1, int(len(vals) * args.thr_quantile))]
            bvals = sorted(it[2]["pf"] for it in fit)
            thr_base = bvals[min(len(bvals) - 1, int(len(bvals) * args.thr_quantile))]
            b = agg_from(pick(ev, lambda it: it[2]["pf"], thr_base))
        else:
            b = agg_topn(ev, lambda it: it[2]["pf"], n, args.per_race)
        for rid, (st, pa) in b.items():
            base_all[rid][0] += st; base_all[rid][1] += pa
        row = {"m": m, "fit": len(fit), "wB1": wB[1], "wF3": wF[3],
               "thr_flow": (thr["flow_tan 単独(単勝プール)"] if use_thr else float("nan")),
               "n": (len(pick(ev, lambda it: it[2]["flow_tan"], thr["flow_tan 単独(単勝プール)"]))
                     if use_thr else
                     (len({i[0] for i in ev}) * args.per_race if args.per_race else n))}
        for nm, sc in variants_for(wB, wF):
            if use_thr:
                _sel = pick(ev, sc, thr[nm])
                a = agg_from(_sel)
            else:
                a = agg_topn(ev, sc, n, args.per_race)
            for rid, (st, pa) in a.items():
                agg_all[nm][rid][0] += st; agg_all[nm][rid][1] += pa
            if use_thr:
                sel_all[nm] += _sel
            elif args.per_race > 0:
                _by = defaultdict(list)
                for _it in ev:
                    _by[_it[0]].append(_it)
                for _rid, _v in _by.items():
                    sel_all[nm] += sorted(_v, key=lambda it: -sc(it))[:args.per_race]
            else:
                sel_all[nm] += sorted(ev, key=lambda it: -sc(it))[:n]
            row[nm] = sum(v[1] for v in a.values()) / sum(v[0] for v in a.values())
        row["base"] = sum(v[1] for v in b.values()) / sum(v[0] for v in b.values())
        per_month.append(row)

    print(f"\n[月別 ROI]  {'月':<8}{'n':>6}{'基準':>8}" + "".join(f"{nm[:10]:>12}" for nm in NAMES)
          + f"{'市場重み':>9}{'flow係数':>9}" + (f"{'flow閾値':>10}" if args.thr_quantile is not None else ""))
    for r in per_month:
        print(f"  {r['m']:<8}{r['n']:>6,}{r['base']:>8.3f}" + "".join(f"{r[nm]:>12.3f}" for nm in NAMES)
              + f"{r['wB1']:>9.3f}{r['wF3']:>9.3f}"
              + (f"{r['thr_flow']:>10.4f}" if args.thr_quantile is not None else ""))
    if args.thr_quantile is not None:
        # 前向き運用用: 手元の全期間を fit とした flow_tan の絶対閾値(hro-ops --flow-threshold に渡す値)。
        # eval 月ごとの閾値(上の列)がこれと大きくズレていないことも確認する(分布の安定性)。
        vals = sorted(it[2]["flow_tan"] for it in items)
        thr_fwd = vals[min(len(vals) - 1, int(len(vals) * args.thr_quantile))]
        n_fwd = sum(1 for v in vals if v >= thr_fwd)
        print(f"\n★ 前向き用 flow_tan 絶対閾値(全期間 {len(items):,} 本の分位 {args.thr_quantile}): "
              f"{thr_fwd:+.4f}  (>=閾値 {n_fwd:,} 本 = {n_fwd/len(items):.1%})"
              f"\n   → hro-ops run-day --strategy flow --flow-threshold {thr_fwd:.4f}")
    if args.dump_selections:
        import csv as _csv
        with open(args.dump_selections, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(["race_id", "umaban", "payout", "odds_snap", "ymd"])
            for rid, pay, ft in sel_all.get(args.dump_variant, []):
                w.writerow([rid, ft["um"], pay, f"{ft['q1']:.1f}", ft["ymd"]])
        print(f"\n採用馬券 {len(sel_all.get(args.dump_variant, [])):,} 件 → {args.dump_selections}"
              f"  (変種: {args.dump_variant})")
    base_all = {k: tuple(v) for k, v in base_all.items()}
    agg_all = {nm: {k: tuple(v) for k, v in d.items()} for nm, d in agg_all.items()}
    report(base_all, agg_all, sel_all, f"全 {len(per_month)} ヶ月をプール")
    wins = {nm: sum(1 for r in per_month if r[nm] > r["base"]) for nm in NAMES}
    print("\n  月別で基準を上回った回数: " + "  ".join(f"{nm[:12]}={wins[nm]}/{len(per_month)}" for nm in NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
