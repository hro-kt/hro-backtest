"""E2E ハーネス: 1レース駆動と期間バックテスト。

データフロー（学習・予測と同一の feat_matrix を使用）:
  build_race_features(online) → score_abilities(win/place モデル) → decide_race(PL)
  → run_decide_pipeline(資金配分=BetOrder) → settlement(nl_hr 払戻突合) → ROI/的中率

接続は3系統（row_factory が異なるため分離）:
  - FeatureDB          : 特徴(online)・レース一覧。dict_row。
  - optimizer.connect  : オッズ load_odds_lookup。dict_row。
  - psycopg(tuple rows): nl_hr load_payout_rows（タプル展開で読むため既定 row factory）。
"""

from __future__ import annotations

from dataclasses import dataclass

from hro_features.config import load_config as load_features_config
from hro_features.db import FeatureDB
from hro_features.online import build_race_features
from hro_features.spec import FEATURE_COLUMNS, feature_schema_hash

from hro_predictor.bundle import ModelBundle
from hro_predictor.predict import score_abilities, TARGET_PLACE, TARGET_WIN

from hro_optimizer.config import BettingConfig, KellyConfig, SimConfig
from hro_optimizer.db import PostgresConfig, connect as opt_connect, load_odds_lookup
from hro_optimizer.io import race_abilities_from_dict

from hro_moneymanager.config import MoneyManagerConfig
from hro_moneymanager.pipeline import run_decide_pipeline

from hro_buyer.models import STATUS_DRY_RUN, ExecutionResult
from hro_buyer.postgres import load_payout_rows
from hro_buyer.settlement import build_payout_index, settle_results, summarize

RaceKey = tuple[str, str, str, str, str, str]


def betting_config(source: str, *, min_er=None, min_prob=None, max_odds_age=None) -> BettingConfig:
    """confirmed は鮮度ガード無制限を既定に（確定オッズは鮮度概念なし）。閾値は任意で上書き。"""
    d = BettingConfig()
    age = max_odds_age
    if age is None:
        age = float("inf") if source == "confirmed" else d.max_odds_age_seconds
    return BettingConfig(
        min_expected_return=(min_er if min_er is not None else d.min_expected_return),
        min_probability=(min_prob if min_prob is not None else d.min_probability),
        max_odds_age_seconds=age,
    )


def load_models(win_path: str, place_path: str) -> tuple[ModelBundle, ModelBundle]:
    """win/place バンドルを読み、現スキーマ一致と役割(target)を検証して返す。"""
    cur_hash = feature_schema_hash()
    win_b = ModelBundle.load(win_path)
    place_b = ModelBundle.load(place_path)
    win_b.meta.assert_compatible(cur_hash, FEATURE_COLUMNS)
    place_b.meta.assert_compatible(cur_hash, FEATURE_COLUMNS)
    if win_b.meta.target != TARGET_WIN:
        raise ValueError(f"--win-model の target は {TARGET_WIN} であるべき: {win_b.meta.target}")
    if place_b.meta.target != TARGET_PLACE:
        raise ValueError(f"--place-model の target は {TARGET_PLACE} であるべき: {place_b.meta.target}")
    return win_b, place_b


def list_races(db: FeatureDB, d_from: str, d_to: str, limit: int | None = None) -> list[RaceKey]:
    """[d_from, d_to](YYYYMMDD) の確定レース(結果あり)のレースキー一覧。"""
    rows = db.query(
        "SELECT DISTINCT year, month_day, jyo_cd, kaiji, nichiji, race_num "
        "FROM feat_labels WHERE (year || month_day) BETWEEN %(a)s AND %(b)s "
        "ORDER BY year, month_day, jyo_cd, race_num",
        {"a": d_from, "b": d_to},
    )
    races = [(r["year"], r["month_day"], r["jyo_cd"], r["kaiji"], r["nichiji"], r["race_num"])
             for r in rows]
    return races[:limit] if limit else races


def orders_for_race(
    db: FeatureDB, conn_odds, win_b: ModelBundle, place_b: ModelBundle, race: RaceKey, *,
    betting: BettingConfig, money: MoneyManagerConfig, sim: SimConfig, kelly: KellyConfig,
    source: str, simultaneous: bool,
) -> tuple[dict | None, list]:
    """1レース: 能力値→PL候補→資金配分。戻り (abilities_dict|None, BetOrder list)。"""
    rows = build_race_features(db, *race)
    if not rows:
        return None, []
    race_id = "".join(race)
    abilities_dict = score_abilities(rows, win_b, place_b, race_id)
    abilities = race_abilities_from_dict(abilities_dict)
    odds_lookup = load_odds_lookup(conn_odds, race, source=source)
    result = run_decide_pipeline(
        abilities, odds_lookup, betting, money,
        sim_config=sim, kelly_config=kelly, simultaneous=simultaneous,
    )
    return abilities_dict, result.orders


def _to_exec(order) -> ExecutionResult:
    """BetOrder → settlement が突合できる ExecutionResult(dry_run 記録扱い)。"""
    return ExecutionResult(
        race_id=order.race_id, selection_id=order.selection_id, bet_type=order.bet_type,
        amount=order.amount, odds=order.odds, mode="dry_run", status=STATUS_DRY_RUN,
        message="backtest",
    )


@dataclass
class BacktestResult:
    n_races: int
    n_races_with_orders: int
    summary: object                  # SettlementSummary（全体）
    by_bet_type: dict                # {bet_type: SettlementSummary}
    settlements: list                # 個別 Settlement（詳細出力・CSV 用）


def run_backtest(
    d_from: str, d_to: str, win_path: str, place_path: str, *,
    source: str = "confirmed", samples: int | None = None, max_total: float | None = None,
    independent_kelly: bool = False, min_er=None, min_prob=None, max_odds_age=None,
    limit: int | None = None, progress=None,
) -> BacktestResult:
    """期間バックテスト: 各レースで bet を生成し、nl_hr 払戻と突合して ROI を出す。

    資金配分はレース独立(store なし)。ROI は比率なので予算スケールに非依存。
    progress(i, n, race_id) を渡すと進捗コールバックされる。
    """
    win_b, place_b = load_models(win_path, place_path)
    betting = betting_config(source, min_er=min_er, min_prob=min_prob, max_odds_age=max_odds_age)
    money = MoneyManagerConfig()
    sim = SimConfig(n_samples=samples) if samples else SimConfig()
    kelly = KellyConfig(max_total=max_total) if max_total else KellyConfig()

    db = FeatureDB(load_features_config())
    conn_odds = opt_connect()
    all_orders: list = []
    n_with = 0
    try:
        races = list_races(db, d_from, d_to, limit=limit)
        for i, race in enumerate(races, 1):
            _, orders = orders_for_race(
                db, conn_odds, win_b, place_b, race,
                betting=betting, money=money, sim=sim, kelly=kelly,
                source=source, simultaneous=not independent_kelly,
            )
            if orders:
                n_with += 1
                all_orders.extend(orders)
            if progress:
                progress(i, len(races), "".join(race))
    finally:
        conn_odds.close()
        db.close()

    # settlement: nl_hr は tuple-row で読む（load_payout_rows がタプル展開のため）
    import psycopg
    results = [_to_exec(o) for o in all_orders]
    race_ids = {o.race_id for o in all_orders}
    conn_hr = psycopg.connect(PostgresConfig.from_env().conninfo)
    try:
        payout_rows = load_payout_rows(conn_hr, race_ids)
    finally:
        conn_hr.close()
    index, settled = build_payout_index(payout_rows)
    settlements = settle_results(results, index, settled)

    by_type = {}
    for bt in ("place", "wide", "trio", "win"):
        sub = [s for s in settlements if s.bet_type == bt]
        if sub:
            by_type[bt] = summarize(sub)

    return BacktestResult(
        n_races=len(races), n_races_with_orders=n_with,
        summary=summarize(settlements), by_bet_type=by_type, settlements=settlements,
    )
