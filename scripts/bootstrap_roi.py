"""候補CSV(--save-candidates出力)から、指定セルのROIをレース単位ブロックブートストラップでCI推定。

同一レース内の複数点は当たり外れが強く相関するため、点単位ではなく**レース単位**で
リサンプルしないとCIが不当に狭くなる(=偶然を有意と誤認する)。

    python scripts/bootstrap_roi.py ~/cand2026_base.csv --bet-type wide --min-er 1.7 --min-prob 0.10

複数窓をプールすると n が増えて CI が縮む(窓は独立なので、単一窓で足りない検出力を稼げる):

    python scripts/bootstrap_roi.py ~/cand2025_base.csv ~/cand2026_base.csv \
        --bet-type wide --min-er 1.7 --min-prob 0.10
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load(path: str, tag: str = "") -> list[tuple]:
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            bt, er, prob, odds, st, hit, pay = row[:7]
            rid = row[9] if len(row) > 9 else ""
            runs = row[7] if len(row) > 7 else ""
            if st != "True":
                continue
            # race_id は窓をまたぐと衝突しうるので、ファイル別にプレフィックスする
            rows.append((bt, float(er), float(prob), float(odds), int(pay), f"{tag}|{rid}",
                         int(runs) if runs not in ("", "None") else None))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+", help="候補CSV(複数指定でプール検定)")
    ap.add_argument("--bet-type", required=True)
    ap.add_argument("--min-er", type=float, default=0.0)
    ap.add_argument("--max-er", type=float, default=None)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--max-prob", type=float, default=None,
                    help="確率上限。--min-prob と組んで確率帯を切る(市場価格を固定した比較用)")
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--min-odds", type=float, default=None,
                    help="オッズ下限。--max-odds と組んでオッズ帯を固定し、その中で確率帯を動かすと"
                         "「市場価格が同じでモデル確率だけ違う」比較になる=市場超えの情報があるかの本質的検定")
    ap.add_argument("--max-career", type=int, default=None,
                    help="馬のキャリア本数(h_n_2y)がこれ以下のみ(新馬=0)。血統が効く土俵の切り出し")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    allrows = [r for i, p in enumerate(args.path) for r in load(p, tag=str(i))]
    sel = [t for t in allrows
           if t[0] == args.bet_type and t[1] >= args.min_er and t[2] >= args.min_prob
           and (args.max_er is None or t[1] < args.max_er)
           and (args.max_prob is None or t[2] < args.max_prob)
           and (args.min_odds is None or t[3] >= args.min_odds)
           and (args.max_odds is None or t[3] <= args.max_odds)
           and (args.max_career is None
                or (t[6] is not None and t[6] <= args.max_career))]
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

    # レース単位に集計してから numpy でベクトル化(1反復ごとに全馬券を舐めると終わらない)
    import numpy as np

    st = np.array([100 * len(r) for r in races], dtype=np.float64)
    pa = np.array([sum(r) for r in races], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    k = len(races)
    out = np.empty(args.iters, dtype=np.float64)
    chunk = max(1, min(200, args.iters))
    done = 0
    while done < args.iters:
        m = min(chunk, args.iters - done)
        idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = pa[idx].sum(1) / st[idx].sum(1)
        done += m
    stats = np.sort(out)
    lo = float(stats[int(0.025 * args.iters)])
    hi = float(stats[int(0.975 * args.iters)])
    p_le1 = float((stats <= 1.0).mean())

    print("  ".join(args.path))
    band = "" if args.min_odds is None and args.max_odds is None else \
        f" odds[{args.min_odds or 0}..{args.max_odds if args.max_odds is not None else '∞'}]"
    print(f"  cell: {args.bet_type} er>={args.min_er}"
          f"{'' if args.max_er is None else f'(<{args.max_er})'} prob>={args.min_prob}"
          f"{'' if args.max_prob is None else f'(<{args.max_prob})'}{band}")
    odds = [t[3] for t in sel]
    print(f"  odds: mean={sum(odds) / len(odds):.2f} "
          f"min={min(odds):.1f} max={max(odds):.1f}   "
          f"(帯を固定した比較では mean が両群で揃っていることを必ず確認する)")
    print(f"  bets={n_bets}  races={k}  hits={hits} ({hits / n_bets:.1%})")
    print(f"  ROI = {roi:.3f}   95%CI [{lo:.3f}, {hi:.3f}]   P(ROI<=1.0) = {p_le1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
