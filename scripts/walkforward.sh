#!/usr/bin/env bash
# 年別ウォークフォワード: test年ごとにモデルを学習し、その年だけをOOS評価して候補を積む。
#
# 目的は2つ。
#  (1) 「確定オッズにedgeがあるか」を n を稼いで検定する。単一窓では
#      wide er>=1.7&prob>=0.10 が 2025 1.131(n431)/2026 1.054(n258) で P(ROI<=1)=0.26/0.44＝判定不能だった。
#  (2) 以後のモデル改善を測る計測器にする。現在のCI半幅0.32ではROI+0.05の改善が見えない。
#
# 各 test 年 Y について:
#   train = 〜(Y-1)/09/30, valid = (Y-1)/10/01〜(Y-1)/12/31, test = Y/01/01〜Y/12/31
# これは既存の 2025窓/2026窓モデルと同じ切り方(valid=直前Q4, test=翌年)。
#
# 使い方(必ず ablation env を先に読む。スキーマ不一致だとfail-fastで落ちる):
#   source ~/prod_env.sh
#   nohup bash scripts/walkforward.sh > ~/wf/driver.log 2>&1 &
#
# 途中で落ちても再実行すれば、出来ているモデル/候補はスキップして続きから走る。
set -uo pipefail

START_YEAR="${START_YEAR:-2020}"
END_YEAR="${END_YEAR:-2026}"
END_DATE_LAST="${END_DATE_LAST:-20260827}"   # 最終年はデータ終端まで
WF_DIR="${WF_DIR:-$HOME/wf}"
BET_TYPES="${BET_TYPES:-wide,place}"         # trio は2窓OOSで棄却済(0.72/0.52)なので既定で外す
# ★収集コストの大半はワイドの組み合わせ(7窓で wide 2,014,574本 vs place 276,588本)。
#   アブレーション比較など複勝だけ見ればよい実験では BET_TYPES=place で大幅に速くなる。
WORKERS="${WORKERS:-3}"
ER_GRID="${ER_GRID:-1.0,1.3,1.5,1.7,2.0}"
PROB_GRID="${PROB_GRID:-0.00,0.05,0.10}"
MAX_ODDS="${MAX_ODDS:-2000}"

# アブレーション未設定のまま回すと、本番と違う特徴スキーマのモデルが黙って出来上がる。
# 事故防止のため既定では拒否し、意図的な全解除は ALLOW_NO_ABLATION=1 で明示させる。
# ★2026-09: ~/prod_env.sh の11フラグ(SED/PEDCOND等)は、evidence-baseline で無効と
#   整理した4年窓/汚染データ時代の証拠で決めたもの。全解除版と比較する実験のために
#   この経路が要る。
if [ -z "${HRO_ABLATE_SED:-}" ] && [ "${ALLOW_NO_ABLATION:-}" != "1" ]; then
  echo "!! ablation env が未設定です。'source ~/prod_env.sh' を先に実行してください。" >&2
  echo "   (特徴スキーマが変わり ModelBundle.assert_compatible で落ちます)" >&2
  echo "   意図的に全特徴で回すなら ALLOW_NO_ABLATION=1 を付けてください。" >&2
  exit 1
fi
if [ "${ALLOW_NO_ABLATION:-}" = "1" ] && [ -z "${HRO_ABLATE_SED:-}" ]; then
  echo "※ アブレーション全解除で実行します(本番11フラグとは別スキーマ)。"
fi

mkdir -p "$WF_DIR"
echo "walkforward: ${START_YEAR}..${END_YEAR}  bet_types=${BET_TYPES}  out=${WF_DIR}"

for Y in $(seq "$START_YEAR" "$END_YEAR"); do
  PREV=$((Y - 1))
  DIR="$WF_DIR/$Y"
  mkdir -p "$DIR"
  VALID_FROM="${PREV}1001"
  TEST_FROM="${Y}0101"
  if [ "$Y" -eq "$END_YEAR" ]; then D_TO="$END_DATE_LAST"; else D_TO="${Y}1231"; fi

  echo "=========================================================="
  echo "[$Y] valid_from=$VALID_FROM test=${TEST_FROM}..${D_TO}"
  date

  for T in win:y_win place:y_fukusyo; do
    NAME="${T%%:*}"; TARGET="${T##*:}"
    OUT="$DIR/${NAME}_prod.joblib"
    if [ -s "$OUT" ]; then echo "[$Y] skip train $NAME (exists)"; continue; fi
    echo "[$Y] train $NAME ($TARGET)"
    if ! poetry run hro-predictor train --target "$TARGET" \
        --valid-from "$VALID_FROM" --test-from "$TEST_FROM" \
        --out "$OUT" > "$DIR/train_${NAME}.log" 2>&1; then
      echo "[$Y] !! train $NAME 失敗。$DIR/train_${NAME}.log を確認して中断" >&2
      tail -20 "$DIR/train_${NAME}.log" >&2
      exit 1
    fi
  done

  CAND="$DIR/cand.csv"
  if [ -s "$CAND" ]; then
    echo "[$Y] skip sweep (candidates exists: $(wc -l < "$CAND") rows)"
  else
    echo "[$Y] sweep ${TEST_FROM}..${D_TO}"
    if ! poetry run hro-backtest sweep \
        --win-model "$DIR/win_prod.joblib" --place-model "$DIR/place_prod.joblib" \
        --from "$TEST_FROM" --to "$D_TO" \
        --bet-types "$BET_TYPES" --max-odds "$MAX_ODDS" --workers "$WORKERS" \
        --er "$ER_GRID" --prob "$PROB_GRID" \
        --save-candidates "$CAND" --out "$DIR/grid.csv" \
        > "$DIR/sweep.log" 2>&1; then
      echo "[$Y] !! sweep 失敗。$DIR/sweep.log を確認して中断" >&2
      tail -20 "$DIR/sweep.log" >&2
      exit 1
    fi
  fi
  echo "[$Y] done: $(wc -l < "$CAND") candidates"
done

echo "=========================================================="
echo "全窓完了。プール検定:"
CANDS=$(ls "$WF_DIR"/*/cand.csv 2>/dev/null | tr '\n' ' ')
echo "  files: $CANDS"
for ER in 1.0 1.3 1.5 1.7 2.0; do
  poetry run python scripts/bootstrap_roi.py $CANDS \
    --bet-type wide --min-er "$ER" --min-prob 0.10 2>&1 | grep -E 'cell:|bets=|ROI ='
done | tee "$WF_DIR/pooled_wide.txt"
date
