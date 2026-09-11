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
    ap.add_argument("--bins", type=int, default=10, help="オッズ分位の分割数(既定10)")
    ap.add_argument("--prob-bins", type=int, default=2,
                    help="各オッズ帯の中で確率を何分位に切るか(既定2=中央値二分)。"
                         "上げると上位分位に絞った時にROIが1.0を越えるかが見える")
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
    # 各オッズ帯の中で確率の分位境界を決める(全データで一度だけ。以降は固定)
    Q = args.prob_bins
    pedges: dict[int, list[float]] = {}
    for b, v in buckets.items():
        ps = sorted(t[0] for t in v)
        pedges[b] = [ps[int(len(ps) * i / Q)] for i in range(1, Q)]

    def qbin_of(b: int, prob: float) -> int:
        q = 0
        for e in pedges[b]:
            if prob < e:
                break
            q += 1
        return min(q, Q - 1)

    med = {b: (pedges[b][Q // 2 - 1] if Q > 1 else median(t[0] for t in v))
           for b, v in buckets.items()}

    # 各点に (bucket, qbin) を付けてレース単位にまとめる
    by_race: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for prob, odds, pay, rid in rows:
        b = bucket_of(odds)
        by_race[rid].append((qbin_of(b, prob), pay))
    races = list(by_race.values())

    print(f"{args.bet_type}: bets={len(rows)} races={len(races)} bins={args.bins}"
          f"{'' if args.min_prob == 0 else f' (prob>={args.min_prob} で事前フィルタ)'}")
    hdr = "".join(f"{f'q{q + 1}':>12}" for q in range(Q))
    print(f"\n各オッズ帯の中を確率で{Q}分位(q1=低prob … q{Q}=高prob)。セル='ROI(n)'")
    print(f"{'odds帯':>16} {'n':>8}{hdr}")
    lo_edge = 0.0
    for b in sorted(buckets):
        v = buckets[b]
        hi_edge = edges[b] if b < len(edges) else float("inf")
        cells = ""
        for q in range(Q):
            g = [t for t in v if qbin_of(b, t[0]) == q]
            r = sum(t[2] for t in g) / (100 * len(g)) if g else float("nan")
            cells += f"{r:>7.3f}({len(g) // 1000:>3}k)" if len(g) >= 1000 else \
                     f"{r:>7.3f}({len(g):>4})"
        print(f"{lo_edge:7.1f}-{hi_edge:7.1f} {len(v):>8,}{cells}")
        lo_edge = hi_edge

    # レース単位に先に集計してから numpy でベクトル化する。
    # 1反復ごとに全馬券を舐めると iters × bets = 28億回になって終わらない。
    import numpy as np

    top = Q - 1
    sh = np.array([sum(100 for q, _ in r if q == top) for r in races], dtype=np.float64)
    ph = np.array([sum(p for q, p in r if q == top) for r in races], dtype=np.float64)
    sl = np.array([sum(100 for q, _ in r if q < top) for r in races], dtype=np.float64)
    pl = np.array([sum(p for q, p in r if q < top) for r in races], dtype=np.float64)

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

    # 最上位分位そのものが 1.0 を越えるか(=買える領域があるか)も同時に出す
    out_h = np.empty(args.iters, dtype=np.float64)
    rng2 = np.random.default_rng(args.seed + 1)
    done = 0
    while done < args.iters:
        m = min(chunk, args.iters - done)
        idx = rng2.integers(0, k, size=(m, k))
        out_h[done:done + m] = ph[idx].sum(1) / sh[idx].sum(1)
        done += m
    hs = np.sort(out_h)
    h_lo, h_hi = float(hs[int(0.025 * args.iters)]), float(hs[int(0.975 * args.iters)])
    p_h_le1 = float((hs <= 1.0).mean())

    print(f"\nプール: ROI(最上位q{Q})={rh:.3f}  ROI(それ以外)={rl:.3f}")
    print(f"  差 = {d:+.3f}   95%CI [{lo_ci:+.3f}, {hi_ci:+.3f}]   P(差<=0) = {p_le0:.3f}")
    print(f"  最上位q{Q}の水準: ROI={rh:.3f}  95%CI [{h_lo:.3f}, {h_hi:.3f}]  "
          f"P(ROI<=1.0)={p_h_le1:.3f}")
    print("\n  差>0 が有意 → 同じ市場価格でモデルは市場より当たりを見分けている(=市場超えの情報あり)。")
    print("  差≒0        → モデルは市場価格を再現しているだけ。prob でROIが動くのは人気馬バイアス。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
