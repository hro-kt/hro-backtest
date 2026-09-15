"""オッズ帯条件の再較正が運用点(prob>=0.40)のROIを動かすかを、候補CSVだけで時系列検証する。

背景(market-referenced slicer, 2026-09): モデルは上位馬を約+2.5pt過信していた
(基準人気1-4: モデル.401 / 市場.376 / 実測.372)。isotonic は valid で当てているのに test で残る。
運用点 prob>=0.40 はまさにその領域なので、市場順位(=オッズ帯)を条件にした再較正で
過信した馬が閾値から落ち、ROI が上がる見込み。ここでは学習・sweep を回さず、
fit窓の候補で「オッズ帯ごとの prob→的中率 の isotonic」を当て、eval窓に適用して測る。

    python scripts/recalibrate_test.py \\
        --fit  ~/wf_noabl/2023/cand.csv ~/wf_noabl/2024/cand.csv \\
        --eval ~/wf_noabl/2025/cand.csv ~/wf_noabl/2026/cand.csv --min-prob 0.40
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

ODDS_EDGES = [1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0]   # 複勝オッズ帯(市場順位の代理)


def load(paths, bet_type):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                bt, _er, prob, odds, st, hit, pay = row[:7]
                if bt != bet_type or st != "True":
                    continue
                out.append((float(prob), float(odds), 1.0 if hit == "True" else 0.0, int(pay),
                            row[9] if len(row) > 9 else ""))
    return out


def band(o):
    return int(np.searchsorted(ODDS_EDGES, o, side="right"))


def pav(x, y):
    """isotonic 回帰(pool-adjacent-violators)。x昇順に並べた y の単調非減少あてはめを返す。"""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x)[order], np.asarray(y, dtype=np.float64)[order]
    # ブロック(値, 重み)
    vals, wts, cnts = [], [], []
    for v in ys:
        vals.append(v); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w; cnts[-2] += cnts[-1]
            vals.pop(); wts.pop(); cnts.pop()
    fitted = np.repeat(vals, cnts)
    # 補間用に (x, fitted) を一意化
    ux, idx = np.unique(xs, return_index=True)
    return ux, fitted[idx]


class Recal:
    def __init__(self, rows, by_band):
        self.by_band = by_band
        self.maps = {}
        groups = defaultdict(list)
        for p, o, h, _pay, _rid in rows:
            groups[band(o) if by_band else 0].append((p, h))
        for b, g in groups.items():
            if len(g) < 200:
                continue
            x = np.array([t[0] for t in g]); y = np.array([t[1] for t in g])
            self.maps[b] = pav(x, y)
        gx = np.array([t[0] for t in rows]); gy = np.array([t[2] for t in rows])
        self.global_map = pav(gx, gy)

    def __call__(self, p, o):
        m = self.maps.get(band(o) if self.by_band else 0) or self.global_map
        return float(np.interp(p, m[0], m[1]))


def race_agg(rows, prob_fn, min_prob, top_n=None):
    scored = [(prob_fn(p, o), pay, rid) for p, o, _h, pay, rid in rows]
    if top_n is not None:
        scored.sort(key=lambda t: -t[0])
        sel = scored[:top_n]
    else:
        sel = [t for t in scored if t[0] >= min_prob]
    agg = defaultdict(lambda: [0.0, 0.0])
    for _q, pay, rid in sel:
        agg[rid][0] += 100.0; agg[rid][1] += pay
    return agg, len(sel)


def paired(aggA, aggB, iters=10000, seed=42):
    rids = sorted(set(aggA) | set(aggB))
    sa = np.array([aggA.get(r, [0, 0])[0] for r in rids]); pa = np.array([aggA.get(r, [0, 0])[1] for r in rids])
    sb = np.array([aggB.get(r, [0, 0])[0] for r in rids]); pb = np.array([aggB.get(r, [0, 0])[1] for r in rids])
    ra, rb = pa.sum() / sa.sum(), pb.sum() / sb.sum()
    rng = np.random.default_rng(seed); k = len(rids); out = np.empty(iters)
    done = 0
    while done < iters:
        m = min(200, iters - done); idx = rng.integers(0, k, size=(m, k))
        out[done:done + m] = pb[idx].sum(1) / sb[idx].sum(1) - pa[idx].sum(1) / sa[idx].sum(1)
        done += m
    d = np.sort(out)
    return ra, rb, rb - ra, float(d[int(0.025 * iters)]), float(d[int(0.975 * iters)]), float((d <= 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--bet-type", default="place")
    ap.add_argument("--min-prob", type=float, default=0.40)
    args = ap.parse_args()

    fit_rows, ev_rows = load(args.fit, args.bet_type), load(args.eval, args.bet_type)
    print(f"fit {len(fit_rows):,}  eval {len(ev_rows):,}  帯: {ODDS_EDGES}")
    raw = lambda p, o: p
    rec_g = Recal(fit_rows, by_band=False)
    rec_b = Recal(fit_rows, by_band=True)

    # 較正の効き(evalの運用点付近): 平均p vs 的中率
    for name, fn in (("生", raw), ("全体再較正", rec_g), ("オッズ帯再較正", rec_b)):
        sel = [(fn(p, o), h) for p, o, h, _pay, _rid in ev_rows if fn(p, o) >= args.min_prob]
        if sel:
            print(f"  {name:<8} prob>={args.min_prob}: n={len(sel):>6,}  平均p={np.mean([s[0] for s in sel]):.3f}"
                  f"  的中率={np.mean([s[1] for s in sel]):.3f}")

    aggR, nR = race_agg(ev_rows, raw, args.min_prob)
    print(f"\n[閾値 prob>={args.min_prob}]  生: n={nR:,}")
    for name, fn in (("全体再較正", rec_g), ("オッズ帯再較正", rec_b)):
        aggX, nX = race_agg(ev_rows, fn, args.min_prob)
        ra, rb, d, lo, hi, p = paired(aggR, aggX)
        print(f"  {name:<8} n={nX:,}  ROI 生={ra:.4f} → {rb:.4f}  差={d:+.4f} [{lo:+.4f},{hi:+.4f}]  P(差<=0)={p:.3f}")

    print(f"\n[同一本数 top-{nR:,}(較正後の確率順)]  ←閾値通過数の変化ではなく順位付けの変化だけを見る")
    for name, fn in (("全体再較正", rec_g), ("オッズ帯再較正", rec_b)):
        aggX, _ = race_agg(ev_rows, fn, args.min_prob, top_n=nR)
        ra, rb, d, lo, hi, p = paired(aggR, aggX)
        print(f"  {name:<8} ROI 生={ra:.4f} → {rb:.4f}  差={d:+.4f} [{lo:+.4f},{hi:+.4f}]  P(差<=0)={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
