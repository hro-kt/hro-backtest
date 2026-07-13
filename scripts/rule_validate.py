"""特徴スライスで見えた強パターンを「除外ルール」として年別に検証する。

feature_slice.py で見えた「systematically 負ける条件」を keep/除外ルール化し、
各年(2023/24/25/26h1)で baseline に対し ROI が改善するかを見る。**全年で改善する頑健な
ルールだけ採用**（cherry-pick と多重検定を避ける）。ルールは経済的に筋が通るもののみ。

使い方(hro-backtest, HRO_ABLATE_SED は学習時と同じ):
    HRO_ABLATE_SED=1 poetry run python scripts/rule_validate.py [wf_cand_dir]
"""

from __future__ import annotations

import csv
import glob
import os
import sys

from hro_features.config import load_config
from hro_features.db import FeatureDB

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser/hro/wf_cand"

FEATCOLS = ["surface", "class_ratio", "distance_m", "field_size",
            "jrdbk_kyakushitsu", "jrdb_shiagari_index", "baba_state"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# keep 条件(True=残す)。None(不明)は保守的に残す。経済的根拠つき。
RULES = {
    "notJump  (障害除外)":       lambda f: str(f.get("surface")) != "障害",
    "notClassUp(昇級除外)":      lambda f: (_num(f.get("class_ratio")) is None
                                           or _num(f.get("class_ratio")) <= 1.0),
    "notLong   (>1800m除外)":    lambda f: (_num(f.get("distance_m")) is None
                                           or _num(f.get("distance_m")) <= 1800),
    "notBigField(>16頭除外)":    lambda f: (_num(f.get("field_size")) is None
                                           or _num(f.get("field_size")) <= 16),
    "notCloser (深追込3除外)":   lambda f: str(f.get("jrdbk_kyakushitsu")) != "3",
    "notPeakFit(仕上>62除外)":   lambda f: (_num(f.get("jrdb_shiagari_index")) is None
                                           or _num(f.get("jrdb_shiagari_index")) <= 62),
}


def _load_bets_by_year(cand_dir):
    by_year = {}
    for fpath in sorted(glob.glob(os.path.join(cand_dir, "place_*.csv"))):
        y = os.path.basename(fpath)[6:-4]
        rows = []
        with open(fpath, encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for x in r:
                if not x or x[0] != "place":
                    continue
                if float(x[1]) < 1.3 or float(x[2]) < 0.15 or x[4] != "True":
                    continue
                rid = x[9] if len(x) > 9 else ""
                sel = x[10] if len(x) > 10 else ""
                if not rid or not sel.strip().isdigit():
                    continue
                rows.append((rid, int(sel), int(x[6])))
        by_year[y] = rows
    return by_year


def _roi(sub):
    n = len(sub)
    return (n, (sum(p for _r, _u, p in sub) / (n * 100)) if n else float("nan"))


def main():
    by_year = _load_bets_by_year(CAND_DIR)
    years = list(by_year)
    all_bets = [b for ys in by_year.values() for b in ys]
    ids = sorted({b[0] for b in all_bets})

    db = FeatureDB(load_config())
    sql = ("SELECT (year||month_day||jyo_cd||kaiji||nichiji||race_num) AS rid, umaban, "
           + ", ".join(FEATCOLS)
           + " FROM feat_matrix "
           "WHERE (year||month_day||jyo_cd||kaiji||nichiji||race_num) = ANY(%(ids)s)")
    feat = {}
    for row in db.query(sql, {"ids": ids}):
        try:
            feat[(row["rid"], int(row["umaban"]))] = row
        except (TypeError, ValueError):
            continue
    db.close()

    def enrich(bets):
        return [(rid, uma, pay, feat.get((rid, uma))) for rid, uma, pay in bets
                if feat.get((rid, uma)) is not None]

    ey = {y: enrich(bs) for y, bs in by_year.items()}

    def roi_year(bets, keep=None):
        sub = [(r, u, p) for r, u, p, f in bets if (keep is None or keep(f))]
        return _roi(sub)

    hdr = "rule".ljust(24) + " | " + " | ".join(y.rjust(12) for y in years) + " |      ALL"
    print(hdr)
    # baseline
    line = "BASELINE".ljust(24)
    allb = []
    for y in years:
        n, r = roi_year(ey[y])
        line += f" | {r:.3f}({n:>4})"
        allb += ey[y]
    n, r = _roi([(a, b, c) for a, b, c, _f in allb])
    print(line + f" | {r:.3f}({n})")

    print("-" * len(hdr))
    for name, keep in RULES.items():
        line = name.ljust(24)
        allk = []
        for y in years:
            n, r = roi_year(ey[y], keep)
            line += f" | {r:.3f}({n:>4})"
            allk += [(a, b, c) for a, b, c, f in ey[y] if keep(f)]
        n, r = _roi(allk)
        print(line + f" | {r:.3f}({n})")

    # 全年でALL改善したルールだけ AND 合成
    print("-" * len(hdr))
    robust = []
    base_all = _roi([(a, b, c) for a, b, c, _f in allb])[1]
    for name, keep in RULES.items():
        ok = all(roi_year(ey[y], keep)[1] >= roi_year(ey[y])[1] for y in years)
        if ok:
            robust.append((name, keep))
    print("全年でbaseline以上のルール:", [n for n, _k in robust] or "なし")
    if robust:
        def keep_all(f):
            return all(k(f) for _n, k in robust)
        allk = [(a, b, c) for y in years for a, b, c, f in ey[y] if keep_all(f)]
        n, r = _roi(allk)
        print(f"合成(頑健ルールAND): ROI={r:.4f} n={n}  (baseline {base_all:.4f})")


if __name__ == "__main__":
    raise SystemExit(main())
