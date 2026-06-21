"""CLI: E2E 1レース駆動(run) と 期間 ROI バックテスト(backtest)。

    # 1レースを predictor→optimizer→moneymanager で一気に流す
    hro-backtest run --win-model models/win.joblib --place-model models/place.joblib \
        --race 2025 0202 05 01 02 11 --source confirmed

    # test 期間の ROI/的中率を nl_hr 払戻と突合して算出
    hro-backtest backtest --from 20250101 --to 20251231 \
        --win-model models/win.joblib --place-model models/place.joblib --source confirmed
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--win-model", required=True, help="単勝モデル(target=y_win)")
    p.add_argument("--place-model", required=True, help="複勝モデル(target=y_fukusyo)")
    p.add_argument("--source", choices=("live", "confirmed"), default="confirmed",
                   help="confirmed=nl_o*(バックテスト) / live=ts_sokuho 最新")
    p.add_argument("--samples", type=int, default=None, help="PL モンテカルロのサンプル数")
    p.add_argument("--max-total", type=float, default=None, help="同時ケリーの Σf 上限")
    p.add_argument("--independent-kelly", action="store_true", help="各馬券独立フルケリー")
    p.add_argument("--min-er", type=float, default=None, help="期待値の下限(既定1.1)")
    p.add_argument("--min-prob", type=float, default=None, help="的中確率の下限(既定0.05)")
    p.add_argument("--max-odds-age", type=float, default=None,
                   help="オッズ鮮度上限(秒)。未指定: confirmed=無制限 / live=60s")


def _cmd_run(args) -> int:
    from hro_features.config import load_config as load_features_config
    from hro_features.db import FeatureDB
    from hro_optimizer.config import KellyConfig, SimConfig
    from hro_optimizer.db import connect as opt_connect
    from hro_moneymanager.config import MoneyManagerConfig
    from . import harness

    win_b, place_b = harness.load_models(args.win_model, args.place_model)
    betting = harness.betting_config(args.source, min_er=args.min_er,
                                     min_prob=args.min_prob, max_odds_age=args.max_odds_age)
    money = MoneyManagerConfig()
    sim = SimConfig(n_samples=args.samples) if args.samples else SimConfig()
    kelly = KellyConfig(max_total=args.max_total) if args.max_total else KellyConfig()

    db = FeatureDB(load_features_config())
    conn = opt_connect()
    try:
        abilities, orders = harness.orders_for_race(
            db, conn, win_b, place_b, tuple(args.race),
            betting=betting, money=money, sim=sim, kelly=kelly,
            source=args.source, simultaneous=not args.independent_kelly,
        )
    finally:
        conn.close()
        db.close()

    if abilities is None:
        print(f"race {''.join(args.race)}: feat_matrix に行がありません（実在レースか確認）")
        return 1
    runners = abilities["runners"]
    sum_win = sum(r["p_win"] for r in runners)
    sum_place = sum(r["p_place"] for r in runners)
    print(f"race_id={abilities['race_id']} field_size={abilities['field_size']} "
          f"Σp_win={sum_win:.3f} Σp_place={sum_place:.3f} source={args.source}")
    print(f"=== BetOrders ({len(orders)}) ===")
    total = 0
    for o in orders:
        total += o.amount
        print(f"  [{o.selection_id}/{o.bet_type}] amount={o.amount:>6}  ER={o.expected_return:.3f} "
              f"edge={o.edge:+.3f}  p={o.probability:.3f}  odds={o.odds}")
    print(f"  total_amount={total} (daily_budget={money.daily_budget})")
    return 0


def _fmt(s) -> str:
    roi = "n/a" if s.roi is None else f"{s.roi:.3f}"
    hit = "n/a" if s.hit_rate is None else f"{s.hit_rate:.3f}"
    return (f"n={s.n} settled={s.n_settled} hit={s.n_hit} "
            f"staked={s.total_staked} payout={s.total_payout} pnl={s.total_pnl:+d} "
            f"ROI={roi} hit_rate={hit}")


def _cmd_backtest(args) -> int:
    from . import harness

    res = harness.run_backtest(
        args.d_from, args.d_to, args.win_model, args.place_model,
        source=args.source, samples=args.samples, max_total=args.max_total,
        independent_kelly=args.independent_kelly, min_er=args.min_er,
        min_prob=args.min_prob, max_odds_age=args.max_odds_age, limit=args.limit,
        show_progress=not args.no_progress,
    )
    print(f"\n=== Backtest [{args.d_from}..{args.d_to}] source={args.source} ===")
    print(f"races={res.n_races} with_orders={res.n_races_with_orders}")
    print(f"[ALL]   {_fmt(res.summary)}")
    for bt, s in res.by_bet_type.items():
        print(f"[{bt:5s}] {_fmt(s)}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["race_id", "selection_id", "bet_type", "amount", "odds",
                        "settled", "hit", "payout", "pnl"])
            for s in res.settlements:
                w.writerow([s.race_id, s.selection_id, s.bet_type, s.amount, s.odds,
                            s.settled, s.hit, s.payout, s.pnl])
        print(f"-> wrote per-bet settlements to {args.out}")
    return 0


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _cmd_sweep(args) -> int:
    from . import harness

    er_grid = _floats(args.er)
    prob_grid = _floats(args.prob)
    bet_types = tuple(x.strip() for x in args.bet_types.split(",") if x.strip())
    settled = harness.collect_settled_candidates(
        args.d_from, args.d_to, args.win_model, args.place_model,
        source=args.source, samples=args.samples, limit=args.limit,
        show_progress=not args.no_progress, bet_types=bet_types,
    )
    # セグメント絞り込み（少キャリア/休み明け＝市場情報が薄い土俵での検証）
    seg = ""
    if args.max_career is not None:
        settled = [t for t in settled if t[7] is not None and t[7] <= args.max_career]
        seg += f" career<={args.max_career}"
    if args.min_layoff is not None:
        settled = [t for t in settled if t[8] is not None and t[8] >= args.min_layoff]
        seg += f" layoff>={args.min_layoff}d"
    table = harness.sweep_roi(settled, er_grid, prob_grid)

    print(f"\n=== Sweep [{args.d_from}..{args.d_to}] source={args.source}{seg} "
          f"(flat ¥100/bet, ROI=payout/stake; cell='ROI(n)') ===")
    for t in ("ALL", "place", "wide", "trio"):
        print(f"\n[{t}]  rows=min_er, cols=min_prob")
        print("  min_er \\ min_prob | " + " | ".join(f"{p:>11.2f}" for p in prob_grid))
        for er in er_grid:
            cells = []
            for p in prob_grid:
                n, roi, _ = table[t][(er, p)]
                cells.append(f"{('--' if roi is None else f'{roi:.3f}'):>6}({n:>4})" if n else f"{'--':>11}")
            print(f"  {er:>16.2f} | " + " | ".join(cells))

    # オッズ帯別 ROI（人気-穴バイアス検証）
    cuts = _floats(args.odds_bands)
    ob, bands = harness.odds_band_roi(settled, cuts, ref_er=args.ref_er, ref_prob=args.ref_prob)
    print(f"\n=== Odds-band ROI  (ref: min_er>={args.ref_er}, min_prob>={args.ref_prob}; cell='ROI(n)') ===")
    for t in ("ALL", "place", "wide", "trio"):
        print(f"\n[{t}]")
        print("  band | " + " | ".join(f"{lo:g}-{hi:g}".rjust(12) for lo, hi in bands))
        cells = []
        for b in bands:
            n, roi, _ = ob[t][b]
            cells.append(f"{('--' if roi is None else f'{roi:.3f}'):>6}({n:>5})" if n else f"{'--':>12}")
        print("  ROI  | " + " | ".join(cells))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bet_type", "min_er", "min_prob", "n", "roi", "hit_rate"])
            for t, cells in table.items():
                for (er, p), (n, roi, hr) in cells.items():
                    w.writerow([t, er, p, n, "" if roi is None else f"{roi:.4f}",
                                "" if hr is None else f"{hr:.4f}"])
        print(f"\n-> wrote full sweep grid to {args.out}")
    return 0


def _cmd_calib(args) -> int:
    from . import harness

    bet_types = tuple(x.strip() for x in args.bet_types.split(",") if x.strip())
    settled = harness.collect_settled_candidates(
        args.d_from, args.d_to, args.win_model, args.place_model,
        source=args.source, samples=args.samples, limit=args.limit,
        show_progress=not args.no_progress, bet_types=bet_types,
    )
    table = harness.calibration_table(settled, bins=args.bins)

    print(f"\n=== Calibration [{args.d_from}..{args.d_to}] source={args.source} "
          f"(settled bets only; deciles of model p) ===")
    print("  予測>実績 が続く=過信(EV水増し)、予測≈実績=較正OK(効率的市場)")
    for t in ("place", "wide", "trio"):
        groups = table.get(t)
        if not groups:
            continue
        # 全体の予測平均 vs 実績平均（グローバル較正）
        n_all = sum(g[1] for g in groups)
        pred_all = sum(g[2] * g[1] for g in groups) / n_all
        act_all = sum(g[3] * g[1] for g in groups) / n_all
        print(f"\n[{t}]  n={n_all}  overall: pred={pred_all:.3f} actual={act_all:.3f} "
              f"(ratio={act_all / pred_all:.2f})" if pred_all else f"\n[{t}]")
        print("  decile |     n | mean_pred |  actual |  ratio |   ROI")
        for b, n, pred, act, roi in groups:
            ratio = (act / pred) if pred else float("nan")
            print(f"   {b:>2}/{args.bins:<2} | {n:>5} |   {pred:.4f} |  {act:.4f} | "
                  f"{ratio:>5.2f}  | {roi:.3f}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bet_type", "decile", "n", "mean_pred", "actual_rate", "roi"])
            for t, groups in table.items():
                for b, n, pred, act, roi in groups:
                    w.writerow([t, b, n, f"{pred:.5f}", f"{act:.5f}", f"{roi:.4f}"])
        print(f"\n-> wrote calibration to {args.out}")
    return 0


def _trio_grid(label, table, er_grid, prob_grid) -> None:
    print(f"\n[trio {label}]  rows=min_er, cols=min_prob")
    print("  min_er \\ min_prob | " + " | ".join(f"{p:>11.2f}" for p in prob_grid))
    for er in er_grid:
        cells = []
        for p in prob_grid:
            n, roi, _ = table[(er, p)]
            cells.append(f"{('--' if roi is None else f'{roi:.3f}'):>6}({n:>4})" if n else f"{'--':>11}")
        print(f"  {er:>16.2f} | " + " | ".join(cells))


def _cmd_trio_calib(args) -> int:
    from . import harness

    raw, cal, pts, er_grid, prob_grid = harness.run_trio_calibration(
        args.cal_from, args.cal_to, args.d_from, args.d_to, args.win_model, args.place_model,
        source=args.source, samples=args.samples, show_progress=not args.no_progress,
    )
    print(f"\n=== Trio calibration  cal=[{args.cal_from}..{args.cal_to}] "
          f"test=[{args.d_from}..{args.d_to}] source={args.source} ===")
    print("  較正マッピング(生PL確率 → 較正後):")
    for x, y in pts:
        print(f"    {x:.4f} -> {y:.4f}")
    _trio_grid("RAW", raw, er_grid, prob_grid)
    _trio_grid("CALIBRATED", cal, er_grid, prob_grid)
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(
        prog="hro-backtest", description="E2E 駆動 & ROI バックテスト(自動購入はしない)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="1レースを predictor→optimizer→moneymanager で実行")
    _add_common(p_run)
    p_run.add_argument("--race", nargs=6, required=True,
                       metavar=("YEAR", "MONTHDAY", "JYO", "KAIJI", "NICHIJI", "RACENUM"))
    p_run.set_defaults(func=_cmd_run)

    p_bt = sub.add_parser("backtest", help="期間の ROI/的中率を nl_hr 突合で算出")
    _add_common(p_bt)
    p_bt.add_argument("--from", dest="d_from", required=True, help="YYYYMMDD")
    p_bt.add_argument("--to", dest="d_to", required=True, help="YYYYMMDD")
    p_bt.add_argument("--limit", type=int, default=None, help="先頭 N レースだけ(動作確認用)")
    p_bt.add_argument("--no-progress", action="store_true", help="進捗バーを表示しない")
    p_bt.add_argument("--out", type=Path, default=None, help="各 bet の突合結果を CSV 出力")
    p_bt.set_defaults(func=_cmd_backtest)

    p_sw = sub.add_parser("sweep", help="min_er×min_prob のグリッドで ROI を一括比較(フラット¥100)")
    p_sw.add_argument("--win-model", required=True)
    p_sw.add_argument("--place-model", required=True)
    p_sw.add_argument("--source", choices=("live", "confirmed"), default="confirmed")
    p_sw.add_argument("--samples", type=int, default=None, help="PL モンテカルロのサンプル数")
    p_sw.add_argument("--from", dest="d_from", required=True, help="YYYYMMDD")
    p_sw.add_argument("--to", dest="d_to", required=True, help="YYYYMMDD")
    p_sw.add_argument("--limit", type=int, default=None, help="先頭 N レースだけ")
    p_sw.add_argument("--er", default="1.0,1.1,1.2,1.3,1.5,2.0", help="min_er グリッド(カンマ区切り)")
    p_sw.add_argument("--prob", default="0.0,0.05,0.10,0.15,0.20,0.25",
                      help="min_prob グリッド(カンマ区切り)")
    p_sw.add_argument("--bet-types", default="place,wide",
                      help="評価する券種(カンマ区切り)。trio はノイズかつ高コストなので既定で除外")
    p_sw.add_argument("--odds-bands", default="1.5,3,6,12,25,50",
                      help="オッズ帯の区切り(カンマ区切り)。人気-穴バイアス検証用")
    p_sw.add_argument("--ref-er", type=float, default=1.0,
                      help="オッズ帯分析の参照 min_er(既定1.0=ほぼ全候補)")
    p_sw.add_argument("--ref-prob", type=float, default=0.0,
                      help="オッズ帯分析の参照 min_prob(既定0.0)")
    p_sw.add_argument("--max-career", type=int, default=None,
                      help="セグメント: 馬券の脚の最少キャリア本数(h_n_2y)がこれ以下のみ。少キャリア検証用")
    p_sw.add_argument("--min-layoff", type=int, default=None,
                      help="セグメント: 最長休養日数がこれ以上のみ。休み明け検証用(新馬=9999)")
    p_sw.add_argument("--no-progress", action="store_true")
    p_sw.add_argument("--out", type=Path, default=None, help="全グリッドを CSV 出力")
    p_sw.set_defaults(func=_cmd_sweep)

    p_cal = sub.add_parser("calib", help="較正診断: モデル確率 vs 実的中率(デシル別)")
    p_cal.add_argument("--win-model", required=True)
    p_cal.add_argument("--place-model", required=True)
    p_cal.add_argument("--source", choices=("live", "confirmed"), default="confirmed")
    p_cal.add_argument("--samples", type=int, default=None)
    p_cal.add_argument("--from", dest="d_from", required=True, help="YYYYMMDD")
    p_cal.add_argument("--to", dest="d_to", required=True, help="YYYYMMDD")
    p_cal.add_argument("--limit", type=int, default=None)
    p_cal.add_argument("--bins", type=int, default=10, help="分位ビン数(既定10=デシル)")
    p_cal.add_argument("--bet-types", default="place,wide",
                       help="評価する券種(カンマ区切り)。trio も見るなら place,wide,trio")
    p_cal.add_argument("--no-progress", action="store_true")
    p_cal.add_argument("--out", type=Path, default=None, help="較正表を CSV 出力")
    p_cal.set_defaults(func=_cmd_calib)

    p_tc = sub.add_parser("trio-calib", help="三連複: PL確率を実頻度で較正→test の trio ROI を raw/較正で比較")
    p_tc.add_argument("--win-model", required=True)
    p_tc.add_argument("--place-model", required=True)
    p_tc.add_argument("--source", choices=("live", "confirmed"), default="confirmed")
    p_tc.add_argument("--samples", type=int, default=None)
    p_tc.add_argument("--cal-from", required=True, help="較正期間 開始 YYYYMMDD(例 20240101)")
    p_tc.add_argument("--cal-to", required=True, help="較正期間 終了 YYYYMMDD(例 20241231)")
    p_tc.add_argument("--from", dest="d_from", required=True, help="test 開始 YYYYMMDD")
    p_tc.add_argument("--to", dest="d_to", required=True, help="test 終了 YYYYMMDD")
    p_tc.add_argument("--no-progress", action="store_true")
    p_tc.set_defaults(func=_cmd_trio_calib)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
