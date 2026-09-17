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

import numpy as np
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
    ap.add_argument("--min-prob", type=float, default=None,
                    help="事前フィルタ(運用帯に絞りたいとき)。既定=無し。★score 列が負になり得る"
                         "(late_flow_csv のフロースコア等)場合に 0.0 を既定にすると負側が全部落ちる")
    ap.add_argument("--score", choices=("model", "market"), default="model",
                    help="market: 確率を 1/odds に置き換える negative control。帯内の順位付けが"
                         "純粋な人気-穴バイアスだけでどれだけの差を生むかを測る(外部レビュー P0)")
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [t for t in load(args.path, args.bet_type)
            if (args.min_prob is None or t[0] >= args.min_prob)
            and (args.max_odds is None or t[1] <= args.max_odds)]
    if args.score == "market":
        # モデル確率を捨て、市場だけの順位付け(1/odds)にする。帯の中でこれが正の差を出すなら、
        # その分は「同一帯内でも人気馬ほどROIが高い」バイアスであり、モデル情報ではない。
        rows = [(1.0 / t[1], t[1], t[2], t[3]) for t in rows]
        print("※ negative control: score = 1/odds(市場のみ)")
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
    # 各オッズ帯の中で確率分位を **順位ベース** で割り当てる(同値はシード付き乱数で分割)。
    # ★閾値方式(prob >= 境界)だと同値が全部同じ側に落ち、帯ごとに上位群が20%からずれる。
    #   複勝オッズは0.1刻みで細い帯では同値だらけなので、1/odds を score にした対照では
    #   帯が細いほど上位群の大きさが帯構成と相関し、プールした ROI差が Simpson 型に膨らんだ
    #   (10帯 +0.039 → 100帯 +0.130 でモデルと一致、という不自然な挙動の原因)。
    #   順位割当なら各帯で厳密に 1/Q ずつになり、プール比較の帯構成が上位/それ以外で揃う。
    Q = args.prob_bins
    rng_tie = np.random.default_rng(args.seed + 7)
    qbin_row: dict[int, int] = {}          # 行index → 分位
    for b, v in buckets.items():
        idxs = [i for i, t in enumerate(rows) if bucket_of(t[1]) == b]
        keys = np.array([rows[i][0] for i in idxs], dtype=np.float64)
        order = np.lexsort((rng_tie.random(len(idxs)), keys))   # score昇順、同値は乱数
        n_b = len(idxs)
        for pos, j in enumerate(order):
            qbin_row[idxs[j]] = min(Q - 1, (pos * Q) // n_b)

    def qbin_of(b: int, prob: float) -> int:  # 表示用の後方互換(帯内の値→分位の近似)
        ps = sorted(t[0] for t in buckets[b])
        k = int(np.searchsorted(ps, prob, side="right")) - 1
        return min(Q - 1, max(0, (k * Q) // max(len(ps), 1)))

    med = {b: median(t[0] for t in v) for b, v in buckets.items()}

    # 各点に (qbin, payout) を付けてレース単位にまとめる
    by_race: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (prob, odds, pay, rid) in enumerate(rows):
        by_race[rid].append((qbin_row[i], pay))
    races = list(by_race.values())

    print(f"{args.bet_type}: bets={len(rows)} races={len(races)} bins={args.bins}"
          f"{'' if args.min_prob is None else f' (prob>={args.min_prob} で事前フィルタ)'}")
    hdr = "".join(f"{f'q{q + 1}':>12}" for q in range(Q))
    print(f"\n各オッズ帯の中を確率で{Q}分位(q1=低prob … q{Q}=高prob)。セル='ROI(n)'")
    print(f"{'odds帯':>16} {'n':>8}{hdr}")
    lo_edge = 0.0
    rows_by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(rows):
        rows_by_bucket[bucket_of(t[1])].append(i)
    for b in sorted(buckets):
        v = buckets[b]
        hi_edge = edges[b] if b < len(edges) else float("inf")
        cells = ""
        for q in range(Q):
            g = [rows[i] for i in rows_by_bucket[b] if qbin_row[i] == q]
            r = sum(t[2] for t in g) / (100 * len(g)) if g else float("nan")
            cells += f"{r:>7.3f}({len(g) // 1000:>3}k)" if len(g) >= 1000 else \
                     f"{r:>7.3f}({len(g):>4})"
        # 帯内の人気-穴バイアス監査: 最上位分位とそれ以外の平均オッズ。差が大きい帯ほど
        # ROI差にバイアスが混ざる。--bins を増やして差が消えるかを見る。
        top = [rows[i][1] for i in rows_by_bucket[b] if qbin_row[i] == Q - 1]
        rest = [rows[i][1] for i in rows_by_bucket[b] if qbin_row[i] < Q - 1]
        mo = (f"  odds top/rest={sum(top)/len(top):.2f}/{sum(rest)/len(rest):.2f}"
              if top and rest else "")
        print(f"{lo_edge:7.1f}-{hi_edge:7.1f} {len(v):>8,}{cells}{mo}")
        lo_edge = hi_edge

    # レース単位に先に集計してから numpy でベクトル化する。
    # 1反復ごとに全馬券を舐めると iters × bets = 28億回になって終わらない。

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

    n_top = int(sh.sum() // 100); n_rest = int(sl.sum() // 100)
    print(f"\nプール: ROI(最上位q{Q})={rh:.3f} (n={n_top:,})  ROI(それ以外)={rl:.3f} (n={n_rest:,})"
          f"   上位比率={n_top / max(n_top + n_rest, 1):.3f} (期待 {1 / Q:.3f})")
    print(f"  差 = {d:+.3f}   95%CI [{lo_ci:+.3f}, {hi_ci:+.3f}]   P(差<=0) = {p_le0:.3f}")
    print(f"  最上位q{Q}の水準: ROI={rh:.3f}  95%CI [{h_lo:.3f}, {h_hi:.3f}]  "
          f"P(ROI<=1.0)={p_h_le1:.3f}")
    print("\n  差>0 が有意 → 同じ市場価格でモデルは市場より当たりを見分けている(=市場超えの情報あり)。")
    print("  差≒0        → モデルは市場価格を再現しているだけ。prob でROIが動くのは人気馬バイアス。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
