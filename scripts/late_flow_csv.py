"""UTMD-149(東大MDC 2026)再現用: 「同じ最終オッズでも、締切直前にオッズが下がった馬の実現returnが高いか」。

ts_o1(公式時系列オッズ 0B41)から、発走 −lead 分時点のスナップショットと確定オッズ(nl_o1)を取り、
直前の資金流入スコアを作って **候補CSV と同じ形式** で書き出す。あとは既存の検定にそのまま流す:

    python scripts/late_flow_csv.py --from 20250901 --to 20260830 --lead-min 5 --out ~/lateflow.csv
    python scripts/market_beat_test.py ~/lateflow.csv --bet-type place --prob-bins 5 --bins 100
      → 最終オッズ帯を固定し、score(直前フロー)の上位/下位で ROI が分かれるか。
        対照: --score market。差>0 なら「価格経路は最終価格に無い情報を持つ」が我々の土俵で再現。

score の定義(prob 列に入れる。大きいほど「直前に買われた」):
  share_i(t) = (1/fuku_odds_low_i(t)) / Σ_j (1/fuku_odds_low_j(t))   … 票数シェアの代理(票数は未取込)
  score_i    = logit(share_i(final)) − logit(share_i(T−lead))          … 直前の相対的な資金流入
  レース内で合計票が増えることの影響はシェア化で消える(レビューの ExcessFlow の相対版)。
odds 列は確定複勝下限(選定に使う値と同じ)、payout は nl_hr の複勝払戻(100円あたり)。
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB

SQL = """
WITH ra AS (
  SELECT year, month_day, jyo_cd, kaiji, nichiji, race_num,
         to_timestamp(year||month_day||hasso_time, 'YYYYMMDDHH24MI') AS post_ts,
         CASE WHEN syusso_tosu ~ '^[0-9]+$' THEN syusso_tosu::int END AS field_size
  FROM nl_ra
  WHERE jyo_cd BETWEEN '01' AND '10' AND year||month_day BETWEEN %(d0)s AND %(d1)s
    AND hasso_time ~ '^[0-9]{4}$'
),
snap AS (  -- 発走 −lead 分 以前の最新スナップショット(馬ごと)
  SELECT DISTINCT ON (t.year, t.month_day, t.jyo_cd, t.kaiji, t.nichiji, t.race_num, t.umaban)
         t.year, t.month_day, t.jyo_cd, t.kaiji, t.nichiji, t.race_num, t.umaban,
         t.fuku_odds_low AS fuku_low_snap, t.hasso_time AS snap_time
  FROM ts_o1 t
  JOIN ra USING (year, month_day, jyo_cd, kaiji, nichiji, race_num)
  WHERE to_timestamp(t.year||t.hasso_time, 'YYYYMMDDHH24MI') <= ra.post_ts - make_interval(mins => %(lead)s)
    AND t.fuku_odds_low ~ '^[0-9]+$' AND t.fuku_odds_low::numeric > 0
  ORDER BY t.year, t.month_day, t.jyo_cd, t.kaiji, t.nichiji, t.race_num, t.umaban, t.hasso_time DESC
)
SELECT s.year, s.month_day, s.jyo_cd, s.kaiji, s.nichiji, s.race_num, s.umaban,
       s.fuku_low_snap::numeric/10.0 AS snap_odds,
       o.fuku_odds_low::numeric/10.0 AS final_odds,
       ra.field_size,
       (SELECT h.pay FROM nl_hr h
         WHERE (h.year,h.month_day,h.jyo_cd,h.kaiji,h.nichiji,h.race_num)
             = (s.year,s.month_day,s.jyo_cd,s.kaiji,s.nichiji,s.race_num)
           AND h.bet_type = 'fuku' AND regexp_replace(h.kumi,'[^0-9]','','g') = s.umaban
         LIMIT 1) AS pay
FROM snap s
JOIN nl_o1 o USING (year, month_day, jyo_cd, kaiji, nichiji, race_num, umaban)
JOIN ra   USING (year, month_day, jyo_cd, kaiji, nichiji, race_num)
WHERE o.fuku_odds_low ~ '^[0-9]+$' AND o.fuku_odds_low::numeric > 0
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", required=True)
    ap.add_argument("--to", dest="d1", required=True)
    ap.add_argument("--lead-min", type=int, default=5, help="発走の何分前のスナップショットと比べるか")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = FeatureDB(load_features_config())
    try:
        rows = db.query(SQL, {"d0": args.d0, "d1": args.d1, "lead": args.lead_min})
    finally:
        db.close()
    if not rows:
        print("行なし(ts_o1 に期間のデータが無いか、nl_ra/nl_o1 と結合できない)")
        return 1

    # レース内シェア(1/odds 正規化)を両時点で
    by_race = defaultdict(list)
    for r in rows:
        by_race[(r["year"], r["month_day"], r["jyo_cd"], r["kaiji"], r["nichiji"], r["race_num"])].append(r)
    out, n_hit = [], 0
    for key, rs in by_race.items():
        s_snap = sum(1.0 / float(r["snap_odds"]) for r in rs)
        s_fin = sum(1.0 / float(r["final_odds"]) for r in rs)
        rid = "".join(key)
        for r in rs:
            sh0 = (1.0 / float(r["snap_odds"])) / s_snap
            sh1 = (1.0 / float(r["final_odds"])) / s_fin
            lg = lambda x: math.log(min(max(x, 1e-6), 1 - 1e-6) / (1 - min(max(x, 1e-6), 1 - 1e-6)))
            score = lg(sh1) - lg(sh0)
            pay = r["pay"]
            pay_i = int(str(pay).strip() or 0) if pay is not None and str(pay).strip().isdigit() else 0
            hit = pay_i > 0
            n_hit += hit
            out.append(["place", round(score, 5), round(score, 5), float(r["final_odds"]), True, hit, pay_i,
                        "", "", rid, r["umaban"]])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bet_type", "er", "prob", "odds", "settled", "hit", "payout",
                    "seg_runs", "seg_layoff", "race_id", "selection_id"])
        w.writerows(out)
    print(f"{len(out):,} 行 / {len(by_race):,} レース / 的中 {n_hit:,} → {args.out}")
    print("  prob 列 = 直前フロースコア(logit share 最終 − T−lead)。odds 列 = 確定複勝下限。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
