#!/usr/bin/env bash
# 既に学習済みの窓(~/wf*/YYYY/*.joblib)を使って、別の券種で候補を再収集する。
#
# walkforward.sh は学習+sweep をセットで回すが、券種を後から足したいだけなら学習は不要。
# 例: 単勝は一度も評価していなかった(収集時の --bet-types に win を入れていなかっただけで、
# モデル・オッズ(nl_o1.tan_odds)・決済(BET_TYPE_TO_HR win->tan)は揃っている)。
#
# ★単勝が安い理由: simulation.py が「単勝は winモデルの p_win をそのまま使う
#   (PL再フィットの影響を受けないクリーンな量)」としているのでモンテカルロを通らない。
#   SAMPLES を小さくしても単勝の確率は一切変わらない。
#
#   source ~/prod_env.sh
#   WF_DIR=~/wf_before_ped BET_TYPES=win SAMPLES=1000 SUFFIX=win \
#     nohup bash scripts/sweep_existing.sh > ~/sweep_win.log 2>&1 &
set -uo pipefail

WF_DIR="${WF_DIR:-$HOME/wf}"
BET_TYPES="${BET_TYPES:-win}"
SUFFIX="${SUFFIX:-$BET_TYPES}"
SAMPLES="${SAMPLES:-1000}"
WORKERS="${WORKERS:-2}"
MAX_ODDS="${MAX_ODDS:-2000}"
ER_GRID="${ER_GRID:-1.0,1.1,1.2,1.3,1.5,1.7,2.0}"
PROB_GRID="${PROB_GRID:-0.00,0.05,0.10,0.20,0.30}"
END_DATE_LAST="${END_DATE_LAST:-20260827}"

if [ -z "${HRO_ABLATE_SED:-}" ]; then
  echo "!! ablation env が未設定です。'source ~/prod_env.sh' を先に実行してください。" >&2
  exit 1
fi

for DIR in "$WF_DIR"/*/; do
  Y=$(basename "$DIR")
  case "$Y" in [0-9][0-9][0-9][0-9]) ;; *) continue ;; esac
  [ -s "$DIR/win_prod.joblib" ] || { echo "[$Y] モデル無し→skip"; continue; }
  OUT="$DIR/cand_${SUFFIX}.csv"
  if [ -s "$OUT" ]; then echo "[$Y] skip (exists)"; continue; fi
  if [ "$Y" -ge 2026 ]; then D_TO="$END_DATE_LAST"; else D_TO="${Y}1231"; fi
  echo "=== [$Y] sweep ${BET_TYPES} ${Y}0101..${D_TO}"; date
  if ! poetry run hro-backtest sweep \
      --win-model "$DIR/win_prod.joblib" --place-model "$DIR/place_prod.joblib" \
      --from "${Y}0101" --to "$D_TO" \
      --bet-types "$BET_TYPES" --max-odds "$MAX_ODDS" --workers "$WORKERS" \
      --samples "$SAMPLES" --er "$ER_GRID" --prob "$PROB_GRID" \
      --save-candidates "$OUT" --out "$DIR/grid_${SUFFIX}.csv" \
      > "$DIR/sweep_${SUFFIX}.log" 2>&1; then
    echo "[$Y] !! 失敗: $DIR/sweep_${SUFFIX}.log" >&2
    tail -20 "$DIR/sweep_${SUFFIX}.log" >&2
    exit 1
  fi
  echo "[$Y] done: $(wc -l < "$OUT") candidates"
done
echo "=== 完了。プール検定:"
poetry run python scripts/bootstrap_roi.py "$WF_DIR"/*/cand_${SUFFIX}.csv \
  --bet-type "${BET_TYPES%%,*}" --min-er 1.0 --min-prob 0.0
date
