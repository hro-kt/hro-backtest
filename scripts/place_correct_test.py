"""複勝の残差補正モデルを walk-forward OOS でオフライン検証する（配管を足す前の可否判定）。

考え方: PL の複勝確率は系統的に過信(約2倍)し、実力/仕上上位を過大・展開/馬場/激走を過小評価する
(feature_slice.py で確認)。そこで **PL複勝確率＋主要特徴 → 実際に複勝したか** を LightGBM で学習し、
補正確率 cal_p を出す。ER = cal_p × odds で選び直す。**各ブロックを直前ブロックのみで学習→次年OOS**、
2024/2025/2026h1 をプールして ROI を baseline(現行 ER>=1.3&prob>=0.15) と比較。

odds はモデル入力に入れない(市場情報=確定オッズを特徴化しない制約)。EV 計算でのみ使う。
公平比較のため、各OOSブロックで **baseline と同じ本数** を cal_ER 上位から選ぶ(volume-matched)。

使い方(hro-backtest, HRO_ABLATE_SED は学習時と同じ):
    HRO_ABLATE_SED=1 poetry run python scripts/place_correct_test.py [wf_cand_dir]
"""

from __future__ import annotations

import csv
import glob
import os
import sys

from hro_features.config import load_config
from hro_features.db import FeatureDB

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/hro/wf_cand"

NUM = ["class_ratio", "race_class", "distance_m", "field_size",
       "jrdb_shiagari_index", "jrdb_oikire_index",
       "jrdbk_gekiso", "jrdbk_idm", "jrdbk_josho", "jrdbk_kijun_ninki",
       "kab_baba_sa", "h_speedidx_6m", "h_speedidx_adj_6m",
       "earnings_cum", "earnings_per_start", "h_fukusyo_6m", "h_fukusyo_1y",
       "h_win_1y", "h_finish_sd_1y", "days_since_prev", "prev_agari_rank"]
CATE = ["surface", "baba_state", "jrdbk_kyakushitsu", "grade_cd", "syubetu_cd"]
ORDER = ["2023", "2024", "2025", "2026h1"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import lightgbm as lgb

    # 1) 全 place 候補(閾値なし)＋baseline該当フラグ を年別に読む
    per_year = {}
    for fp in glob.glob(os.path.join(CAND_DIR, "place_*.csv")):
        y = os.path.basename(fp)[6:-4]
        rows = []
        with open(fp, encoding="utf-8") as fh:
            r = csv.reader(fh); next(r, None)
            for x in r:
                if not x or x[0] != "place" or x[4] != "True":
                    continue
                rid = x[9] if len(x) > 9 else ""
                sel = x[10] if len(x) > 10 else ""
                if not rid or not sel.strip().isdigit():
                    continue
                er, prob, odds = float(x[1]), float(x[2]), float(x[3])
                base = (er >= 1.3 and prob >= 0.15)
                rows.append(dict(rid=rid, uma=int(sel), prob=prob, odds=odds,
                                 hit=1 if x[5] == "True" else 0, pay=int(x[6]), base=base))
        per_year[y] = rows

    # 2) 特徴を join
    ids = sorted({r["rid"] for rows in per_year.values() for r in rows})
    db = FeatureDB(load_config())
    sql = ("SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) AS rid, umaban, "
           + ", ".join(NUM + CATE)
           + " FROM feat_matrix WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num)=ANY(%(ids)s)")
    feat = {}
    for row in db.query(sql, {"ids": ids}):
        try:
            feat[(row["rid"], int(row["umaban"]))] = row
        except (TypeError, ValueError):
            continue
    db.close()

    # カテゴリを整数コード化(全データでfit)
    codes = {c: {} for c in CATE}
    def code(c, v):
        d = codes[c]; k = str(v)
        if k not in d:
            d[k] = len(d)
        return d[k]

    def vec(r):
        fr = feat.get((r["rid"], r["uma"]))
        if fr is None:
            return None
        x = [r["prob"]] + [_f(fr.get(c)) for c in NUM] + [code(c, fr.get(c)) for c in CATE]
        return x

    for rows in per_year.values():
        for r in rows:
            r["x"] = vec(r)
    for y in per_year:
        per_year[y] = [r for r in per_year[y] if r["x"] is not None]

    ncat = len(CATE)
    cat_idx = list(range(1 + len(NUM), 1 + len(NUM) + ncat))

    def build(rows):
        X = np.array([r["x"] for r in rows], dtype=float)
        yv = np.array([r["hit"] for r in rows], dtype=int)
        return X, yv

    def roi(sel):
        n = len(sel)
        return (n, sum(r["pay"] for r in sel) / (n * 100) if n else float("nan"),
                100 * sum(r["hit"] for r in sel) / n if n else 0)

    # 3) walk-forward: block i を それ以前で学習した補正で OOS 選別
    print(f"{'block':8} {'baseline ROI(n)':>20} {'corrected ROI(n)':>22} {'hit%':>7}")
    pooled_base, pooled_corr, pooled_flo, pooled_fhi = [], [], [], []
    for i in range(1, len(ORDER)):
        te = ORDER[i]
        if te not in per_year:
            continue
        train_rows = [r for j in range(i) for r in per_year.get(ORDER[j], [])]
        if len(train_rows) < 200:
            continue
        Xtr, ytr = build(train_rows)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                 num_leaves=31, min_child_samples=50,
                                 subsample=0.8, colsample_bytree=0.8, verbosity=-1)
        clf.fit(Xtr, ytr, categorical_feature=cat_idx)
        te_rows = per_year[te]
        Xte, _ = build(te_rows)
        calp = clf.predict_proba(Xte)[:, 1]
        for r, p in zip(te_rows, calp):
            r["cal_p"] = p
            r["cal_er"] = p * r["odds"]
        base_sel = [r for r in te_rows if r["base"]]
        k = len(base_sel)
        corr_sel = sorted(te_rows, key=lambda r: -r["cal_er"])[:k]  # volume-matched
        nb, rb, _ = roi(base_sel)
        nc, rc, hc = roi(corr_sel)
        print(f"{te:8} {f'{rb:.3f}({nb})':>20} {f'{rc:.3f}({nc})':>22} {hc:>6.1f}%")
        pooled_base += base_sel; pooled_corr += corr_sel
        # 変種: baseline内を cal_p 中央値で二分(補正=フィルタ)。上位半分が良ければ低cal_pを落とせる。
        bs = sorted(base_sel, key=lambda r: r["cal_p"])
        h = len(bs) // 2
        pooled_flo += bs[:h]; pooled_fhi += bs[h:]

    nb, rb, _ = roi(pooled_base)
    nc, rc, hc = roi(pooled_corr)
    print("-" * 60)
    print(f"POOLED baseline:  ROI={rb:.4f} n={nb}")
    print(f"POOLED corrected(再選別): ROI={rc:.4f} n={nc} hit={hc:.1f}%")
    nlo, rlo, _ = roi(pooled_flo)
    nhi, rhi, _ = roi(pooled_fhi)
    print(f"[フィルタ変種] baseline内 cal_p 下位半分: ROI={rlo:.4f} n={nlo}")
    print(f"[フィルタ変種] baseline内 cal_p 上位半分: ROI={rhi:.4f} n={nhi}")
    print("  → 上位>>下位 なら『低cal_pのbaseline betを落とす』が有効(補正=フィルタ)")


if __name__ == "__main__":
    raise SystemExit(main())
