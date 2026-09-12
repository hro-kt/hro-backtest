"""2つの候補セット(=変更前/変更後)を、同一レース上で対応をつけて比較する。

なぜ必要か:
  変更前後は同じレースを走っているので、それぞれの周辺CIを見比べても差の精度を
  過小評価する(共通の年変動が両方に乗っているため)。同じレース集合をリサンプルして
  差を直接ブートストラップすれば、共通変動が打ち消えて検出力が上がる。

指標は market_beat_test と同じ「オッズ帯の中の最上位確率分位のROI」。オッズ帯の境界は
A側で決めて両者に適用する(帯を揃えないと比較にならない)。確率分位は各側で独自に切る
(各モデルの上位20%を選ぶ、という意味にするため)。

    python scripts/paired_compare.py \\
        --a ~/wf_before_ped/2020/cand.csv ~/wf_before_ped/2021/cand.csv \\
        --b ~/wf/2020/cand.csv ~/wf/2021/cand.csv --bet-type place

    # 素のセル(er/prob閾値)で比べたいとき
    python scripts/paired_compare.py --a ... --b ... --bet-type place --cell --min-prob 0.40
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np


def load(paths: list[str], bet_type: str) -> list[tuple]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                bt, er, prob, odds, st, _hit, pay = row[:7]
                if bt != bet_type or st != "True":
                    continue
                rows.append((float(prob), float(odds), int(pay),
                             row[9] if len(row) > 9 else "", float(er)))
    return rows


def select(rows, edges, q, cell, min_er, min_prob):
    """各行を採用/不採用に分ける。返り値: {race_id: (stake, payout)}"""
    if cell:
        keep = [t for t in rows if t[4] >= min_er and t[0] >= min_prob]
    else:
        def bucket_of(o):
            b = 0
            for e in edges:
                if o < e:
                    break
                b += 1
            return min(b, len(edges))
        buckets = defaultdict(list)
        for t in rows:
            buckets[bucket_of(t[1])].append(t)
        keep = []
        for b, v in buckets.items():
            ps = sorted(x[0] for x in v)
            thr = ps[int(len(ps) * (q - 1) / q)]   # 上位 1/q 分位の下限
            keep += [x for x in v if x[0] >= thr]
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for _p, _o, pay, rid, _er in keep:
        agg[rid][0] += 100
        agg[rid][1] += pay
    return {k: tuple(v) for k, v in agg.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="変更前の候補CSV")
    ap.add_argument("--b", nargs="+", required=True, help="変更後の候補CSV")
    ap.add_argument("--bet-type", required=True)
    ap.add_argument("--odds-bins", type=int, default=10)
    ap.add_argument("--prob-bins", type=int, default=5, help="最上位1/Nを採用")
    ap.add_argument("--cell", action="store_true", help="分位でなく er/prob 閾値で選ぶ")
    ap.add_argument("--min-er", type=float, default=0.0)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    A, B = load(args.a, args.bet_type), load(args.b, args.bet_type)
    if not A or not B:
        print(f"データ不足 A={len(A)} B={len(B)}")
        return 1
    # オッズ帯の境界は A 側で決めて両者に適用する(帯を揃えないと比較にならない)
    srt = sorted(t[1] for t in A)
    edges = [srt[int(len(srt) * i / args.odds_bins)] for i in range(1, args.odds_bins)]

    sa = select(A, edges, args.prob_bins, args.cell, args.min_er, args.min_prob)
    sb = select(B, edges, args.prob_bins, args.cell, args.min_er, args.min_prob)
    shared = sorted(set(sa) & set(sb))
    if len(shared) < 100:
        print(f"共通レースが少なすぎます: {len(shared)}")
        return 1

    st_a = np.array([sa[r][0] for r in shared], dtype=np.float64)
    pa_a = np.array([sa[r][1] for r in shared], dtype=np.float64)
    st_b = np.array([sb[r][0] for r in shared], dtype=np.float64)
    pa_b = np.array([sb[r][1] for r in shared], dtype=np.float64)

    roi_a = pa_a.sum() / st_a.sum()
    roi_b = pa_b.sum() / st_b.sum()

    rng = np.random.default_rng(args.seed)
    k = len(shared)
    out = np.empty(args.iters, dtype=np.float64)
    chunk = max(1, min(200, args.iters))
    done = 0
    while done < args.iters:
        m = min(chunk, args.iters - done)
        idx = rng.integers(0, k, size=(m, k))   # 同じレース添字を両側に使う=対応のある比較
        out[done:done + m] = (pa_b[idx].sum(1) / st_b[idx].sum(1)
                              - pa_a[idx].sum(1) / st_a[idx].sum(1))
        done += m
    d = np.sort(out)
    lo, hi = float(d[int(0.025 * args.iters)]), float(d[int(0.975 * args.iters)])
    p_le0 = float((d <= 0.0).mean())

    mode = (f"cell er>={args.min_er} prob>={args.min_prob}" if args.cell
            else f"オッズ{args.odds_bins}分位の中の最上位1/{args.prob_bins}分位")
    print(f"{args.bet_type}  選別={mode}")
    print(f"  共通レース={k:,}   A(前) 本数={int(st_a.sum() // 100):,}  "
          f"B(後) 本数={int(st_b.sum() // 100):,}")
    print(f"  ROI  A(前)={roi_a:.4f}   B(後)={roi_b:.4f}")
    print(f"  差(後-前) = {roi_b - roi_a:+.4f}   95%CI [{lo:+.4f}, {hi:+.4f}]   "
          f"P(差<=0) = {p_le0:.3f}")
    print("\n  対応のある比較なので共通の年変動が打ち消える。周辺CIより狭くなるのが正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
