# META: Memory-Enhanced Trading Agent

META combines multi-agent technical analysis with episodic memory to make LONG/SHORT decisions informed by past trading experiences.

![META memory system](assets/META_memory_flow.png)
![META workflow](assets/META_overview_(1).png)

META integrates **Perception**, **Synthesis**, and **Memory**. Specialized signal agents analyze current market conditions, while the decision agent combines their reports with historical experiences retrieved through cosine similarity over concatenated OHLCV and technical-indicator vectors. Each experience records market conditions, analysis, decisions, outcomes, and reflections. Post-trade reflection guides the addition, revision, removal, or retention of memories, enabling past lessons to inform future decisions.

![META main results](assets/META_Main_Results.png)

In the paper's evaluation across CL, ES, NQ, QQQ, and BTC, META achieves the highest directional accuracy on four of five assets against QuantAgent and a random baseline. It reaches **64.0% on ES** and **62.0% on NQ**, exceeding QuantAgent by **9.0** and **8.7 percentage points**, respectively. Return-based performance varies across assets and metrics.

## Quick start

Use Python 3.11 or 3.12. Run the following from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Test the pipeline without API calls.
python run_graph.py --csv examples/synthetic_ohlcv.csv \
  --output runs/demo --offline --steps 2
```

Offline mode uses synthetic data and deterministic model responses to verify the pipeline; it does not measure predictive performance.

For live model calls, set `OPENAI_API_KEY` in your shell environment. Defaults are `gpt-4o-mini` for chart-tool requests and `gpt-4o` for analysis, decisions, reflection, and memory management.

```bash
python run_graph.py --csv path/to/market.csv \
  --symbol BTC --time-frame 4h --window 90 --horizon 3 \
  --output runs/btc_sample --steps 3
```

CSV files should contain `Datetime,Open,High,Low,Close,Volume` in strictly increasing timestamp order, with positive prices and volume. Single-file mode requires at least `window + horizon` rows and a window of at least 70.

## Benchmark evaluation

Arrange samples as `benchmark/btc/BTC_4h_1.csv`, `benchmark/btc/BTC_4h_2.csv`, etc. Benchmark data must be supplied separately.

```bash
# Evaluate every CSV in one asset folder.
python run_graph.py --benchmark-dir benchmark/btc \
  --output runs/btc_benchmark

# Evaluate multiple assets with separate memory stores.
python run_graph.py --benchmark-root benchmark \
  --assets btc cl es nq qqq --output runs/meta_benchmarks
```

Add `--steps 3` for a small trial per asset. Folder mode makes one decision per CSV, holding out the final three candles by default (`--horizon 3`); each sample needs at least 70 observed candles. Samples are processed by decision timestamp, and memories become available only after their outcomes are observable.

**Use a new, empty output directory for every run.** Each run starts a fresh memory base. Outputs include reports, decisions, reflections, and retrieval counts in `results.jsonl`, retained experiences in `memories.jsonl`, generated charts, and a `summary.json`. Folder-mode outputs are grouped by asset.

The runner evaluates close-to-close directional returns without transaction costs or order execution; its summaries do not reproduce every metric in the paper's results table.

## Configuration

Use `--config path/to/config.json` to override settings in `default_config.py`. Indicator-based retrieval is the default, selecting up to **5 memories** with cosine similarity of at least **0.7**. Use `--mode semantic` to compare analysis-text embeddings instead; this requires embedding API access.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

Built on [QuantAgent](https://github.com/Y-Research-SBU/QuantAgent).
