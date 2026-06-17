"""hro-backtest: E2E ドライバ & ROI バックテスト。

predictor(能力値) → optimizer(PL候補) → moneymanager(資金配分) → settlement(nl_hr 払戻突合)
を一気通貫で回し、回収率(ROI)・的中率を「お金の単位」で評価する。
"""
