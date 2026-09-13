#!/usr/bin/env bash
# ウォークフォワードの進捗を1画面で出す。
#
# ログが3段(ablation.log → driver.log → 窓ごとの train/sweep ログ)に入れ子になっていて
# 外側を tail しても何も動かないため、状態をまとめて見るためのもの。
#
#   bash scripts/wf_status.sh                    # ~/wf_feat ~/wf_noabl
#   bash scripts/wf_status.sh ~/wf ~/wf_feat
set -uo pipefail
DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("$HOME/wf_feat" "$HOME/wf_noabl")

for D in "${DIRS[@]}"; do
  [ -d "$D" ] || { echo "— $D : 未作成"; continue; }
  LOG="$D/driver.log"
  echo "=================================================="
  echo "$D"
  [ -f "$LOG" ] || { echo "  driver.log なし(未起動)"; continue; }

  # driver.log から「窓の開始時刻」を拾う: '[YYYY] valid_from=...' の次行が date 出力
  mapfile -t STARTS < <(awk '/^\[[0-9]{4}\] valid_from=/{y=substr($1,2,4); getline d; print y"\t"d}' "$LOG")
  prev_y=""; prev_t=""; total=0; cnt=0
  for row in "${STARTS[@]}"; do
    y="${row%%$'\t'*}"; t="${row#*$'\t'}"
    ts=$(date -d "$t" +%s 2>/dev/null) || ts=""
    if [ -n "$prev_t" ] && [ -n "$ts" ]; then
      dur=$(( ts - prev_t )); total=$((total+dur)); cnt=$((cnt+1))
      printf "  %s  完了 %dh%02dm\n" "$prev_y" $((dur/3600)) $(((dur%3600)/60))
    fi
    prev_y="$y"; prev_t="$ts"
  done

  # 進行中の窓
  if [ -n "$prev_y" ]; then
    now=$(date +%s); el=$(( now - prev_t ))
    if [ -s "$D/$prev_y/cand.csv" ]; then
      printf "  %s  完了 %dh%02dm (最終)\n" "$prev_y" $((el/3600)) $(((el%3600)/60))
      RUNNING=""
    else
      printf "  %s  ← 進行中 %dh%02dm経過\n" "$prev_y" $((el/3600)) $(((el%3600)/60))
      RUNNING="$prev_y"
    fi
  fi

  if [ "$cnt" -gt 0 ]; then
    avg=$(( total / cnt ))
    printf "  1窓平均 %dh%02dm\n" $((avg/3600)) $(((avg%3600)/60))
    # 残り窓数 = driver.log 冒頭の範囲から算出
    rng=$(sed -n 's/^walkforward: \([0-9]*\)\.\.\([0-9]*\) .*/\1 \2/p' "$LOG" | head -1)
    if [ -n "$rng" ]; then
      set -- $rng; lo=$1; hi=$2
      left=$(( hi - ${RUNNING:-$prev_y} ))
      [ -n "${RUNNING:-}" ] && left=$((left+1))
      printf "  残り %d窓 → 概算 %dh%02dm\n" "$left" $(( (left*avg)/3600 )) $(( ((left*avg)%3600)/60 ))
    fi
  fi

  if [ -n "${RUNNING:-}" ]; then
    for f in "$D/$RUNNING/sweep.log" "$D/$RUNNING/train_place.log" "$D/$RUNNING/train_win.log"; do
      [ -s "$f" ] || continue
      echo "  --- $(basename "$f") 末尾:"
      tail -c 400 "$f" | tr '\r' '\n' | grep -v '^$' | tail -2 | sed 's/^/      /'
      break
    done
  fi
done
