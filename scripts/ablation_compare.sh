#!/usr/bin/env bash
# 本番アブレーション有り / 無し の2本をウォークフォワードで回して対応のある比較にかける。
#
# ★動機: ~/prod_env.sh は11フラグ
#   (SED PEDCOND RACESTRUCT SEASON TRIP GROUNDLOSS TYBODDS PACE TRAJ SOS FIELDSHAPE)
#   を立てて特徴群を除外しているが、その判断は evidence-baseline で無効と整理した
#   4年窓・汚染データ時代の証拠に基づく。特に SED は有料購読している JRDB の成績素点。
#   11の特徴群を失効した理由で切ったまま「情報が足りない」と判断し続けていた。
#
# 券種は place のみ。収集コストの大半はワイドの組み合わせ(7窓で wide 2,014,574本 vs
# place 276,588本)で、複勝が最良の土俵と確定しているため。
#
#   cd ~/hro-backtest
#   nohup bash scripts/ablation_compare.sh > ~/ablation.log 2>&1 &
set -uo pipefail

PROD_ENV="${PROD_ENV:-$HOME/prod_env.sh}"
WF_A="${WF_A:-$HOME/wf_feat}"      # 本番アブレーション有り
WF_B="${WF_B:-$HOME/wf_noabl}"     # 全解除
START_YEAR="${START_YEAR:-2019}"
END_YEAR="${END_YEAR:-2026}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -f "$PROD_ENV" ] || { echo "!! $PROD_ENV が無い" >&2; exit 1; }
mkdir -p "$WF_A" "$WF_B"

echo "=== [A] 本番アブレーション有り → $WF_A"; date
bash -c "source '$PROD_ENV'
  WF_DIR='$WF_A' BET_TYPES=place START_YEAR='$START_YEAR' END_YEAR='$END_YEAR' \
    bash '$HERE/walkforward.sh'" > "$WF_A/driver.log" 2>&1
rc=$?
[ $rc -eq 0 ] || { echo "!! A が失敗($rc)。$WF_A/driver.log" >&2; tail -20 "$WF_A/driver.log" >&2; exit 1; }

echo "=== [B] アブレーション全解除 → $WF_B"; date
# 親シェルに HRO_ABLATE_* が残っていても確実に外す
bash -c "for v in \$(env | sed -n 's/^\(HRO_ABLATE_[A-Z]*\)=.*/\1/p'); do unset \"\$v\"; done
  ALLOW_NO_ABLATION=1 WF_DIR='$WF_B' BET_TYPES=place START_YEAR='$START_YEAR' END_YEAR='$END_YEAR' \
    bash '$HERE/walkforward.sh'" > "$WF_B/driver.log" 2>&1
rc=$?
[ $rc -eq 0 ] || { echo "!! B が失敗($rc)。$WF_B/driver.log" >&2; tail -20 "$WF_B/driver.log" >&2; exit 1; }

echo "=== 対応のある比較 (A=アブレーション有り, B=全解除)"; date
A=$(ls "$WF_A"/*/cand.csv 2>/dev/null)
B=$(ls "$WF_B"/*/cand.csv 2>/dev/null)
for opt in "--cell --min-prob 0.40" "" "--cell --min-prob 0.20 --max-career 0"; do
  # shellcheck disable=SC2086
  poetry run python "$HERE/paired_compare.py" --a $A --b $B --bet-type place $opt
done
date
