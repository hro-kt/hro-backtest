"""モデルが市場価格を超える情報を持つかの検定。

やること: 候補をオッズ十分位に分け、各帯の中で**モデル確率の中央値で二分**して
高確率群と低確率群のROIを比べる。オッズ(=市場価格)を揃えているので、差が出れば
「同じ値段の馬券について、モデルは市場より当たりを見分けられている」ことになる。

なぜこれが必要か:
  prob を上げるとROIが上がる(place 0.879→0.935, wide 0.864→0.910)が、prob が高い＝
  オッズが低いので、人気馬-穴馬バイアス(市場側の性質)と区別できない。オッズを固定すれば
  切り分けられる。er = prob × odds なので、オッズ固定下では prob の順位 = er の順位＝
  er の検定も同時にやり直したことになる。

中央値は全データで一度だけ決め、ブートストラップ中は固定する(再選択による上振れを避ける)。
リサンプルはレース単位(同一レース内の点は強く相関する)。

    python scripts/market_beat_test.py ~/wf/*/cand.csv --bet-type place
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import median


def load(paths: list[str], bet_type: str) -> list[tuple]:
    rows = []
    for i, path in enumerate(paths):
        with open(path, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                bt, _er, prob, odds, st, _hit, pay = row[:7]
                if bt != bet_type or st != "True":
                    continue
                rid = row[9] if len(row) > 9 else ""
                rows.append((float(prob), float(odds), int(pay), f"{i}|{rid}"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="+")
    ap.add_argument("--bet-type", required=True)
    ap.add_argument("--bins", type=int, default=10, help="オッズ十分位の分割数(既定10)")
    ap.add_argument("--min-prob", type=float, default=0.0, help="事前フィルタ(運用帯に絞りたいとき)")
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [t for t in load(args.path, args.bet_type)
            if t[0] >= args.min_prob and (args.max_odds is None or t[1] <= args.max_odds)]
    if len(rows) < 200:
        print(f"データ不足: {len(rows)}")
        return 1

    # オッズ分位の境界(全データで一度だけ決める)
    srt = sorted(t[1] for t in rows)
    edges = [srt[int(len(srt) * i / args.bins)] for i in range(1, args.bins)]

    def bucket_of(odds: float) -> int:
        b = 0
        for e in edges:
            if odds < e:
                break
            b += 1
        return min(b, args.bins - 1)

    buckets: dict[int, list[tuple]] = defaultdict(list)
    for t in rows:
        buckets[bucket_of(t[1])].append(t)
    med = {b: median(t[0] for t in v) for b, v in buckets.items()}

    # 各点に (bucket, is_high) を付けてレース単位にまとめる
    by_race: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    for prob, odds, pay, rid in rows:
        b = bucket_of(odds)
        by_race[rid].append((prob >= med[b], pay))
    races = list(by_race.values())

    print(f"{args.bet_type}: bets={len(rows)} races={len(races)} bins={args.bins}"
          f"{'' if args.min_prob == 0 else f' (prob>={args.min_prob} で事前フィルタ)'}")
    print(f"\n{'odds帯':>16} {'n':>7} {'p中央値':>8} {'ROI高prob':>10} {'ROI低prob':>10} {'差':>8}")
    lo_edge = 0.0
    for b in sorted(buckets):
        v = buckets[b]
        hi = [t for t in v if t[0] >= med[b]]
        lo = [t for t in v if t[0] < med[b]]
        hi_edge = edges[b] if b < len(edges) else float("inf")
        r_hi = sum(t[2] for t in hi) / (100 * len(hi)) if hi else float("nan")
        r_lo = sum(t[2] for t in lo) / (100 * len(lo)) if lo else float("nan")
        print(f"{lo_edge:7.1f}-{hi_edge:7.1f} {len(v):>7,} {med[b]:>8.3f} "
              f"{r_hi:>10.3f} {r_lo:>10.3f} {r_hi - r_lo:>+8.3f}")
        lo_edge = hi_edge

    # レース単位に先に集計してから numpy でベクトル化する。
    # 1反復ごとに全馬券を舐めると iters × bets = 28億回になって終わらない。
    import numpy as np

    sh = np.array([sum(100 for h, _ in r if h) for r in races], dtype=np.float64)
    ph = np.array([sum(p for h, p in r if h) for r in races], dtype=np.float64)
    sl = np.array([sum(100 for h, _ in r if not h) for r in races], dtype=np.float64)
    pl = np.array([sum(p for h, p in r if not h) for r in races], dtype=np.float64)

    rh = ph.sum() / sh.sum()
    rl = pl.sum() / sl.sum()
    d = rh - rl

    rng = np.random.default_rng(args.seed)
    k = len(races)
    out = np.empty(args.iters, dtype=np.float64)
    chunk = max(1, min(200, args.iters))   # (chunk × k) の添字行列に収まる粒度
    done = 0
    while done < args.iters:
        m = min(chunk, args.iters - done)
        idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = (ph[idx].sum(1) / sh[idx].sum(1)
                              - pl[idx].sum(1) / sl[idx].sum(1))
        done += m
    diffs = np.sort(out)
    lo_ci = float(diffs[int(0.025 * args.iters)])
    hi_ci = float(diffs[int(0.975 * args.iters)])
    p_le0 = float((diffs <= 0.0).mean())

    print(f"\nプール: ROI(高prob)={rh:.3f}  ROI(低prob)={rl:.3f}")
    print(f"  差 = {d:+.3f}   95%CI [{lo_ci:+.3f}, {hi_ci:+.3f}]   P(差<=0) = {p_le0:.3f}")
    print("\n  差>0 が有意 → 同じ市場価格でモデルは市場より当たりを見分けている(=市場超えの情報あり)。")
    print("  差≒0        → モデルは市場価格を再現しているだけ。prob でROIが動くのは人気馬バイアス。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
