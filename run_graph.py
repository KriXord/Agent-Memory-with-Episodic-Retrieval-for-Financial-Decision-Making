"""Research runner for continuous CSVs or complete benchmark folders. Never submits orders."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from trading_graph import TradingGraph

COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

def load_data(path):
    data = pd.read_csv(path)
    aliases = {"datetime": "Datetime", "timestamp": "Datetime", "time": "Datetime",
               "date": "Datetime", **{c.lower(): c for c in COLUMNS[1:]}}
    data = data.rename(columns={c: aliases.get(c.strip().lower(), c) for c in data.columns})
    if data.columns.duplicated().any():
        raise ValueError("Ambiguous duplicate CSV columns after alias normalization")
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    data = data[COLUMNS].copy()
    raw_dates = data["Datetime"]
    if pd.api.types.is_numeric_dtype(raw_dates):
        # Seconds vs milliseconds for modern market datasets.
        unit = "ms" if raw_dates.abs().median() >= 1e11 else "s"
        dates = pd.to_datetime(raw_dates, unit=unit, utc=True, errors="raise")
    else:
        dates = pd.to_datetime(raw_dates, utc=True, errors="raise")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Datetime must be nonmissing, strictly increasing and unique")
    data["Datetime"] = dates.dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    values = data[COLUMNS[1:]].to_numpy()
    if not np.isfinite(values).all() or (data["Volume"] <= 0).any():
        raise ValueError("OHLCV must be finite and volume positive")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any() or (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Inconsistent OHLC candles")
    return data

@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def run(csv, output, symbol="DEMO", time_frame="4h", window=90, horizon=3,
        steps=None, offline=False, mode="indicator", config=None):
    if window < 70 or horizon < 1 or (steps is not None and steps < 1):
        raise ValueError("Use window >= 70, horizon >= 1, steps >= 1")
    data = load_data(csv)
    if len(data) < window + horizon:
        raise ValueError("CSV does not contain a full observation window and outcome horizon")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory; reusing future memories invalidates evaluation")
    output.mkdir(parents=True, exist_ok=True)
    memory_file = output / "memories.jsonl"
    run_config = {**(config or {}), "horizon": horizon, "retrieval_mode": mode}
    if offline and mode == "semantic":
        raise ValueError("Offline CLI uses indicator retrieval; semantic retrieval requires embeddings")
    if offline:
        from offline_model import OfflineModel
        graph = TradingGraph(run_config, agent_llm=OfflineModel(), graph_llm=OfflineModel())
    else:
        graph = TradingGraph(run_config)
    (output / "config.json").write_text(json.dumps({
        "symbol": symbol, "time_frame": time_frame, "window": window,
        "offline": offline, "csv": str(Path(csv).resolve()), **graph.config}, indent=2))
    results = []
    # Once the outcome at t+h is consolidated, the next decision occurs at t+h.
    # No decision at t+1 is allowed to access an outcome observed only at t+h.
    for index in range(window - 1, len(data) - horizon, horizon):
        if steps is not None and len(results) >= steps:
            break
        observed = data.iloc[index-window+1:index+1].to_dict("list")
        full = data.iloc[index-window+1:index+horizon+1].to_dict("list")
        chart_dir = output / "charts" / f"step_{len(results):05d}"
        chart_dir.mkdir(parents=True)
        state = {"kline_data": observed, "orig_kline_data": full, "stock_name": symbol,
                 "time_frame": time_frame, "memory_file": str(memory_file)}
        with working_directory(chart_dir):
            final = graph.invoke(state)
        outcome = final["trade_outcome_details"]
        row = {"timestamp": observed["Datetime"][-1],
               "outcome_timestamp": full["Datetime"][-1],
               "decision": final["final_trade_decision"], "pnl": final["trade_outcome"],
               "return": final["trade_outcome"] / outcome["entry_price"],
               "outcome": outcome, "analysis": final["analysis_results"],
               "reports": final["report_dict"], "reflection": final["reflection"],
               "retrieved_memory_count": final["retrieved_memory_count"],
               "memory_actions": final["memory_actions"], "memory_count": final["memory_count"]}
        with (output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        results.append(row)
        print(f"Step {len(results)}: {row['decision']} pnl={row['pnl']:.4f}, memories={row['memory_count']}")
    summary = {"steps": len(results), "offline": offline,
               "positive_pnl_fraction": float(np.mean([r["pnl"] > 0 for r in results])),
               "mean_signed_return": float(np.mean([r["return"] for r in results])),
               "note": "Directional research simulation; no costs, position sizing, or order execution."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="One continuous CSV")
    source.add_argument("--benchmark-dir", help="One asset folder, e.g. benchmark/btc")
    source.add_argument("--benchmark-root", help="Root containing asset subfolders")
    parser.add_argument("--assets", nargs="+", help="Asset subfolders under --benchmark-root; default: all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", help="Optional label override for one asset")
    parser.add_argument("--time-frame", help="Override interval; folder mode infers it from filenames")
    parser.add_argument("--window", type=int, help="CSV sliding window (default 90); folder mode uses all rows except horizon")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mode", choices=["indicator", "semantic"], default="indicator")
    parser.add_argument("--config", help="Optional JSON config for model and memory settings")
    args = vars(parser.parse_args())
    if args["config"]:
        args["config"] = json.loads(Path(args["config"]).read_text())
    benchmark_dir, benchmark_root, assets = (args.pop("benchmark_dir"), args.pop("benchmark_root"), args.pop("assets"))
    if benchmark_dir or benchmark_root:
        if args.pop("window") is not None:
            parser.error("--window is only for --csv; folder samples use all observed rows")
        args.pop("csv")
        from benchmark_runner import run_benchmark
        run_benchmark(benchmark_dir=benchmark_dir, benchmark_root=benchmark_root, assets=assets, **args)
    else:
        if assets:
            parser.error("--assets requires --benchmark-root")
        args["window"] = args["window"] or 90
        args["symbol"] = args["symbol"] or "DEMO"
        args["time_frame"] = args["time_frame"] or "4h"
        run(**args)

if __name__ == "__main__":
    main()
