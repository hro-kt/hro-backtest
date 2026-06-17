# hro-backtest

E2E ドライバ ＆ ROI バックテスト。**predictor(能力値) → optimizer(PL候補) → moneymanager(資金配分)
→ settlement(nl_hr 払戻突合)** を一気通貫で回し、**回収率(ROI)・的中率**を「お金の単位」で評価する。

点指標(AUC/top-k)では測れない「実際に儲かるか」を見るための評価基盤。特徴追加・閾値調整の
良し悪しはここで判断する。**自動購入は一切しない**（dry-run 記録扱いで突合するのみ）。

## 依存

`hro-predictor`(→features) と `hro-buyer`(→moneymanager→optimizer) を editable path 依存で取り込む。
学習・予測と同一の `feat_matrix` / 同一モデルバンドルを使うので、評価が本番と一致する。

```bash
cd hro-backtest
poetry install      # psycopg[binary]/numpy/上流一式が入る
```

接続情報はリポジトリ直下 `.env`（`POSTGRES_*`）を自動ロード。

## 使い方

```bash
# 1レースを一気通貫（predictor→optimizer→moneymanager）。Σp_win/Σp_place と BetOrder を表示
poetry run hro-backtest run \
  --win-model ../hro-predictor/models/win.joblib \
  --place-model ../hro-predictor/models/place.joblib \
  --race 2025 0202 05 01 02 11 --source confirmed

# test 期間の ROI/的中率を nl_hr 払戻と突合して算出（券種別内訳つき）
poetry run hro-backtest backtest --from 20250101 --to 20251231 \
  --win-model ../hro-predictor/models/win.joblib \
  --place-model ../hro-predictor/models/place.joblib \
  --source confirmed --out settlements.csv

# まず動作確認（先頭50レースだけ・サンプル少なめで高速に）
poetry run hro-backtest backtest --from 20250101 --to 20251231 \
  --win-model ... --place-model ... --limit 50 --samples 2000
```

主なフラグ: `--source`(confirmed/live)、`--min-er`/`--min-prob`/`--max-odds-age`(候補フィルタ)、
`--samples`(PL MC 数)、`--max-total`/`--independent-kelly`(ケリー)、`--limit`(レース数上限)、`--out`(CSV)。

## 出力の見方

- `[ALL]` 全体、続いて `[place]/[wide]/[trio]` 券種別に: `n`(賭けた数)/`settled`(確定数)/`hit`(的中数)/
  `staked`(投資)/`payout`(払戻)/`pnl`(損益)/**`ROI`=払戻÷投資**/**`hit_rate`=的中÷確定**。
- **ROI > 1.0 で期待プラス**。特徴追加や閾値変更の前後でこの数字を比較する。

## 注意

- 資金配分はレース独立（日次予算 state なし）。ROI は比率なので予算スケールに非依存。
- `--source confirmed` は確定オッズ(締切後の最終値)での評価。厳密には締切時点 live スナップショットで
  評価すべき（最終オッズは結果情報を含む）。傾向把握には十分だが、本番運用は live で。
- nl_hr は的中組合せ行のみ → 払戻行が1件も無いレースは「未確定」として損益に算入しない。
