"""レース条件のセグメント別に ROI と信頼区間を出す。

狙い: 市場の効率は条件によって違うはず。少頭数戦や下級条件は投票額が小さく情報も薄いので、
そこだけモデルが控除率を越えている可能性がある。オッズ帯では既に切って何も出なかったが
(どの帯も大nでは0.80〜0.85に張り付く)、レース条件では切っていない。

候補CSVの race_id から feat_matrix のレース属性を引いて分類し、各セグメントで
レース単位ブロックブートストラップの CI を出す。P(ROI<=1.0) が小さいセグメントを探す。

    python scripts/segment_roi.py ~/wf_before_ped/*/cand.csv --bet-type place --min-prob 0.40
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB

RID = "(year||month_day||jyo_cd||kaiji||nichiji||race_num)"
WANT = ["field_size", "surface", "distance_m", "baba_state", "race_class",
        "jyo_cd", "race_month"]


def load(paths, bet_type, min_er, min_prob, max_odds):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                bt, er, prob, odds, st, _hit, pay = row[:7]
                if bt != bet_type or st != "True":
                    continue
                if float(er) < min_er or float(prob) < min_prob:
                    continue
                if max_odds is not None and float(odds) > max_odds:
                    continue
                rows.append((int(pay), row[9] if len(row) > 9 else ""))
    return rows


def race_attrs(race_ids):
    db = FeatureDB(load_features_config())
    try:
        with db.conn.cursor() as cur:
            cur.execute("""SELECT a.attname c FROM pg_attribute a JOIN pg_class k ON k.oid=a.attrelid
                           WHERE k.relname='feat_matrix' AND a.attnum>0 AND NOT a.attisdropped""")
            have = {r["c"] for r in cur.fetchall()}
        cols = [c for c in WANT if c in have]
        sel = ", ".join(f"max({c}) {c}" for c in cols)
        out, ids = {}, list({r for r in race_ids if r})
        for i in range(0, len(ids), 5000):
            for r in db.query(f"SELECT {RID} rid, {sel} FROM feat_matrix "
                              f"WHERE {RID} = ANY(%(ids)s) GROUP BY 1",
                              {"ids": ids[i:i + 5000]}):
                out[r["rid"]] = r
        return out, cols
    finally:
        db.close()


def bucket(col, v, cuts):
    if v is None:
        return "unknown"
    if col == "field_size":
        return "<=8" if v <= 8 else "9-12" if v <= 12 else "13-15" if v <= 15 else ">=16"
    if col == "distance_m":
        return ("sprint" if v < 1400 else "mile" if v < 1800
                else "middle" if v < 2200 else "long")
    if col == "baba_state":
        return "soft" if str(v) in ("2", "3", "4") else "firm"
    if col == "race_class":
        q = cuts.get("race_class")
        if not q:
            return "unknown"
        return ("Q1(低)" if v <= q[0] else "Q2" if v <= q[1]
                else "Q3" if v <= q[2] else "Q4(高)")
    return str(v)


def boot(races, iters, seed):
    st = np.array([100 * len(v) for v in races], dtype=np.float64)
    pa = np.array([sum(v) for v in races], dtype=np.float64)
    rng = np.random.default_rng(seed)
    k = len(races)
    out = np.empty(iters, dtype=np.float64)
    chunk, done = max(1, min(200, iters)), 0
    while done < iters:
        m = min(chunk, iters - done)
        idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = pa[idx].sum(1) / st[idx].sum(1)
        done += m
    o = np.sort(out)
    return (pa.sum() / st.sum(), float(o[int(0.025 * iters)]),
            float(o[int(0.975 * iters)]), float((o <= 1.0).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+")
    ap.add_argument("--bet-type", required=True)
    ap.add_argument("--min-er", type=float, default=0.0)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--min-races", type=int, default=200, help="これ未満のセグメントは表示しない")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load(args.path, args.bet_type, args.min_er, args.min_prob, args.max_odds)
    if not rows:
        print("該当なし")
        return 1
    attrs, cols = race_attrs([r[1] for r in rows])
    cuts = {}
    if "race_class" in cols:
        vals = sorted(v["race_class"] for v in attrs.values() if v["race_class"] is not None)
        if vals:
            cuts["race_class"] = [vals[len(vals) * i // 4] for i in (1, 2, 3)]

    print(f"{args.bet_type}  選別: er>={args.min_er} prob>={args.min_prob}  "
          f"馬券={len(rows):,}  レース={len({r[1] for r in rows}):,}")
    all_races = defaultdict(list)
    for pay, rid in rows:
        all_races[rid].append(pay)
    roi, lo, hi, p1 = boot(list(all_races.values()), args.iters, args.seed)
    print(f"  全体: ROI={roi:.3f} [{lo:.3f}, {hi:.3f}] P(ROI<=1)={p1:.3f}\n")

    for col in cols:
        groups = defaultdict(lambda: defaultdict(list))
        for pay, rid in rows:
            a = attrs.get(rid)
            if a is None:
                continue
            groups[bucket(col, a[col], cuts)][rid].append(pay)
        shown = [(k, v) for k, v in groups.items() if len(v) >= args.min_races]
        if not shown:
            continue
        print(f"[{col}]")
        for key, races in sorted(shown):
            roi, lo, hi, p1 = boot(list(races.values()), args.iters, args.seed)
            flag = "  ★" if p1 < 0.10 else ""
            print(f"  {key:<10} races={len(races):>6,} ROI={roi:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] P(ROI<=1)={p1:.3f}{flag}")
        print()
    print("★ = P(ROI<=1.0) < 0.10。セグメント数だけ検定しているので、"
          "出たら必ず別窓で再現を確認すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
