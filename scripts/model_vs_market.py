"""特徴ビンごとに「モデルp / 市場含意p / 実測」を並べ、市場との食い違いで誰が正しいかを見る。

★なぜ「予測 vs 実測」ではなく「モデル vs 市場 vs 実測」なのか:
  hro-predictor diagnose(予測vs実測のギャップ)は max|gap| 0.041 で、モデルは実測に対して
  よく較正されている。しかし市場価格を特徴に入れて AUC/logloss を全窓で改善させたら ROI は
  下がった(2026-09)。賭けの成績を決めるのは絶対精度ではなく**市場との直交成分**。
  だから診断の参照点は市場でなければならない。

  食い違ってモデルが正しいビン = 市場が織り込めていない領域(賭ける/強める特徴を作る)
  食い違って市場が正しいビン  = モデルの盲点(直す特徴/足す交互作用)

入力: sweep --save-candidates の候補CSV(place)。prob=モデルp, odds=確定複勝オッズ,
      市場含意p = 0.80/odds(複勝控除率20%を戻す。粗いが比較の基準としては十分)。
特徴: --features 指定、無ければ --bundle の top_importance(gain上位)。
      race_id + 馬番で feat_matrix に結合して取る。

    python scripts/model_vs_market.py ~/wf_noabl/*/cand.csv \\
        --bundle ~/wf_noabl/2026/place_prod.joblib --top 12
    python scripts/model_vs_market.py ~/wf_noabl/*/cand.csv --features h_n_2y,field_size,tyb_padoku_idx
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB

RID = "(year||month_day||jyo_cd||kaiji||nichiji||race_num)"
TAKEOUT_PLACE = 0.20


def load(paths, bet_type):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                bt, _er, prob, odds, st, hit, pay = row[:7]
                if bt != bet_type or st != "True":
                    continue
                rid = row[9] if len(row) > 9 else ""
                sel = row[10] if len(row) > 10 else ""
                if not rid or not sel.strip().isdigit():
                    continue
                rows.append((rid, int(sel), float(prob), float(odds), hit == "True", int(pay)))
    return rows


def top_features(bundle_path, n):
    from hro_predictor.bundle import ModelBundle
    b = ModelBundle.load(bundle_path)
    imp = (b.meta.metrics or {}).get("top_importance") or {}
    return list(imp)[:n]


def fetch_features(pairs, cols):
    """(race_id, umaban) → {col: value}。feat_matrix から5000レース単位で引く。"""
    db = FeatureDB(load_features_config())
    try:
        with db.conn.cursor() as cur:
            cur.execute("""SELECT a.attname c FROM pg_attribute a JOIN pg_class k ON k.oid=a.attrelid
                           WHERE k.relname='feat_matrix' AND a.attnum>0 AND NOT a.attisdropped""")
            have = {r["c"] for r in cur.fetchall()}
        cols = [c for c in cols if c in have]
        missing = [c for c in cols if c not in have]
        if missing:
            print(f"  (feat_matrix に無い列は除外: {missing})")
        rids = sorted({p[0] for p in pairs})
        sel = ", ".join(cols)
        out = {}
        for i in range(0, len(rids), 5000):
            for r in db.query(
                f"SELECT {RID} rid, umaban_no um, {sel} FROM feat_matrix "
                f"WHERE {RID} = ANY(%(ids)s)", {"ids": rids[i:i + 5000]}):
                out[(r["rid"], r["um"])] = r
        return out, cols
    finally:
        db.close()


def bins_for(values, max_cat=12, q=8):
    """値→ビンラベル関数。低カーディナリティは値そのまま、数値は分位。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return lambda v: "NA", []
    distinct = set(vals)
    if len(distinct) <= max_cat:
        order = sorted(distinct, key=lambda x: (isinstance(x, str), x))
        return (lambda v: "NA" if v is None else str(v)), [str(x) for x in order]
    try:
        arr = np.array([float(v) for v in vals], dtype=np.float64)
    except (TypeError, ValueError):
        return (lambda v: "NA" if v is None else str(v)), None
    edges = np.unique(np.quantile(arr, np.linspace(0, 1, q + 1)[1:-1]))
    labels = []
    lo = -np.inf
    for e in list(edges) + [np.inf]:
        labels.append(f"{lo:.3g}..{e:.3g}" if np.isfinite(lo) or np.isfinite(e) else "all")
        lo = e

    def f(v):
        if v is None:
            return "NA"
        x = float(v)
        k = int(np.searchsorted(edges, x, side="right"))
        return labels[k]
    return f, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+")
    ap.add_argument("--bet-type", default="place")
    ap.add_argument("--bundle", default=None, help="top_importance を取るバンドル(.joblib)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--features", default=None, help="カンマ区切り(指定時は --bundle 不要)")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=300, help="これ未満のビンは表示しない")
    ap.add_argument("--min-prob", type=float, default=0.0, help="モデルpの下限で事前に絞る")
    args = ap.parse_args()

    rows = load(args.path, args.bet_type)
    if args.min_prob > 0:
        rows = [r for r in rows if r[2] >= args.min_prob]
    if not rows:
        print("候補なし")
        return 1
    feats = ([c.strip() for c in args.features.split(",") if c.strip()] if args.features
             else top_features(args.bundle, args.top) if args.bundle else None)
    if not feats:
        ap.error("--features か --bundle を指定してください")

    attrs, feats = fetch_features([(r[0], r[1]) for r in rows], feats)
    joined = [(r, attrs.get((r[0], r[1]))) for r in rows]
    joined = [(r, a) for r, a in joined if a is not None]
    print(f"{args.bet_type}: 候補 {len(rows):,} → feat_matrix 結合 {len(joined):,}"
          f"  (市場含意p = {1 - TAKEOUT_PLACE:.2f}/odds)")

    def agg(items):
        n = len(items)
        pm = float(np.mean([r[2] for r, _ in items]))
        pk = float(np.mean([(1 - TAKEOUT_PLACE) / r[3] for r, _ in items]))
        hit = float(np.mean([1.0 if r[4] else 0.0 for r, _ in items]))
        roi = sum(r[5] for r, _ in items) / (100.0 * n)
        return n, pm, pk, hit, roi

    summary = []
    for c in feats:
        f, order = bins_for([a[c] for _, a in joined], q=args.bins)
        groups = defaultdict(list)
        for r, a in joined:
            groups[f(a[c])].append((r, a))
        keys = [k for k in (order or sorted(groups)) if k in groups] + (["NA"] if "NA" in groups and "NA" not in (order or []) else [])
        print(f"\n[{c}]")
        print(f"  {'bin':<18}{'n':>7}{'モデルp':>9}{'市場p':>8}{'実測':>8}{'ROI':>7}   モデル−市場  実測−市場  判定")
        model_right = market_right = 0
        for k in keys:
            items = groups[k]
            if len(items) < args.min_n:
                continue
            n, pm, pk, hit, roi = agg(items)
            dm, dr = pm - pk, hit - pk
            if abs(dm) < 0.01:
                verdict = "一致"
            elif dm * dr > 0 and abs(dr) >= abs(dm) * 0.5:
                verdict = "★モデル正"; model_right += n
            elif dm * dr <= 0:
                verdict = "市場正"; market_right += n
            else:
                verdict = "微妙"
            print(f"  {k:<18}{n:>7,}{pm:>9.3f}{pk:>8.3f}{hit:>8.3f}{roi:>7.3f}   {dm:+.3f}      {dr:+.3f}     {verdict}")
        tot = model_right + market_right
        summary.append((c, model_right, market_right, tot))

    print("\n=== 特徴別: 食い違いビンで誰が正しかったか(n加重) ===")
    for c, mr, kr, tot in sorted(summary, key=lambda x: -x[1]):
        if tot == 0:
            print(f"  {c:<28} 食い違いビンなし")
        else:
            print(f"  {c:<28} モデル正 {mr:>7,}  市場正 {kr:>7,}  モデル側比率 {mr / tot:.2f}")
    print("\n  ★モデル正 が集中するビン = 市場が織り込めていない領域。市場正 = モデルの盲点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
