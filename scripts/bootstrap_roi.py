"""候補CSV(--save-candidates出力)から、指定セルのROIをレース単位ブロックブートストラップでCI推定。

同一レース内の複数点は当たり外れが強く相関するため、点単位ではなく**レース単位**で
リサンプルしないとCIが不当に狭くなる(=偶然を有意と誤認する)。

    python scripts/bootstrap_roi.py ~/cand2026_base.csv --bet-type wide --min-er 1.7 --min-prob 0.10
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict


def load(path: str) -> list[tuple]:
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            bt, er, prob, odds, st, hit, pay = row[:7]
            rid = row[9] if len(row) > 9 else ""
            if st != "True":
                continue
            rows.append((bt, float(er), float(prob), float(odds), int(pay), rid))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--bet-type", required=True)
    ap.add_argument("--min-er", type=float, default=0.0)
    ap.add_argument("--max-er", type=float, default=None)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sel = [t for t in load(args.path)
           if t[0] == args.bet_type and t[1] >= args.min_er and t[2] >= args.min_prob
           and (args.max_er is None or t[1] < args.max_er)
           and (args.max_odds is None or t[3] <= args.max_odds)]
    if not sel:
        print("該当なし")
        return 1

    by_race: dict[str, list[int]] = defaultdict(list)
    for _bt, _er, _p, _o, pay, rid in sel:
        by_race[rid].append(pay)
    races = list(by_race.values())
    n_bets = len(sel)
    hits = sum(1 for t in sel if t[4] > 0)
    stake = 100 * n_bets
    roi = sum(t[4] for t in sel) / stake

    rnd = random.Random(args.seed)
    k = len(races)
    stats = []
    for _ in range(args.iters):
        s = p = 0
        for _ in range(k):
            r = races[rnd.randrange(k)]
            s += 100 * len(r)
            p += sum(r)
        stats.append(p / s)
    stats.sort()
    lo = stats[int(0.025 * args.iters)]
    hi = stats[int(0.975 * args.iters)]
    p_le1 = sum(1 for x in stats if x <= 1.0) / args.iters

    print(f"{args.path}")
    print(f"  cell: {args.bet_type} er>={args.min_er}"
          f"{'' if args.max_er is None else f'(<{args.max_er})'} prob>={args.min_prob}")
    print(f"  bets={n_bets}  races={k}  hits={hits} ({hits / n_bets:.1%})")
    print(f"  ROI = {roi:.3f}   95%CI [{lo:.3f}, {hi:.3f}]   P(ROI<=1.0) = {p_le1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
