# Reconstruction record

## Evidence and scope

The implementation uses the eleven supplied source/README files, two partial memory implementations, the recovered `UPDATE_TRADING_MEMORY_PROMPT`, the inline VWAP agent example, and the latest supplied paper. The user's clarification defines indicator retrieval as the concatenation of all technical-indicator vectors and normalized OHLCV vectors. The paper's experimental setup confirms k=5 and threshold=0.7; its retrieval-modality discussion also describes technical features plus OHLCV.

## Provenance by component

| File | Origin and changes |
|---|---|
| `graph_util.py` | Supplied numerical and chart code retained. Constant vectors normalize to zero; nonfinite values fail. Charts reduced from 600 to 120 DPI; timestamp parsing repaired; incomplete/negative outcome indices rejected; MACD legend repair. |
| `decision_agent.py` | Supplied decision and reflection prompts retained. Fibonacci state lookup repaired, constructor configured, retrieval mode configurable, malformed decisions fail rather than fabricate an outcome, horizon made explicit, entry index corrected, obsolete commented implementation/debug prints removed. |
| `agent_state.py` | Supplied state expanded with configuration, reports, outcome details, and memory statistics. Symbol type corrected to string. |
| `strategy_agents/{indicator,pattern,trend}_agent.py` | Supplied code and prompts retained, with prompt invocation/message protocol and chart-tool-name repairs. Chart tool requests are required. |
| Eight additional chart-agent modules | Reconstructed strategy prompts and wrappers following the supplied VWAP workflow. Original wording was not recovered. |
| `strategy_agents/vwap_agent.py` | Reconstructed from the supplied inline example through the common helper; not a byte-for-byte restoration. |
| `memory.py` | Reconstructed by combining the supplied dataclass, cosine retrieval, persistence, and event-application design. Compatible with existing constructor and retrieval calls; reflection and optional semantic embeddings added. |
| `memory_prompts.py` | Supplied management prompt preserved as a constant. New adapter adds an explicit ID contract matching the event parser. |
| `memory_agent.py` | Reconstructed post-trade node expected by `create_memory_agent(llm)`. Uses surviving reports, outcomes, and reflections. The older retrieval-only node is unnecessary because the decision agent already retrieves. |
| `graph_setup.py`, `trading_graph.py` | Rebuilt orchestration preserving sequential order and public class names. Unconnected ToolNodes removed; each agent executes its own tool calls, as in the source. |
| Runner, offline model, tests, docs | Newly implemented. Synthetic CSV is newly generated and not experimental data. |

## Explicit reconstruction choices

- Key layout retains the surviving utility order: anchored VWAP; HA close/high/low/open; MACD/signal/histogram; ROC; RSI; short/long SMA; stochastic K/D; Williams %R; then O/H/L/C/V. 20 series x 30 points = 600. Plain VWAP, Bollinger, and Fibonacci are not extra vector blocks because the supplied vector builder did not include them.
- Per-series normalization remains enabled. Optional population-level feature scaling defaults OFF; older snippets defaulted it ON. This choice is configurable.
- Default capacity is 10,000, with oldest insertion pruned when necessary. This is a supplied prototype default, not independently verified as an experiment setting.
- JSONL replaces the earlier pickle and JSON-container prototypes. There is no automatic import/migration of pickle files or unknown legacy schemas.
- Memory analysis stores the full analyst report collection. Decision rationale is stored in metadata. Semantic retrieval embeds the report collection in deterministic key order. No outcome or reflection is embedded into the current query.
- The embedding model defaults to `text-embedding-3-small` as a configurable reconstruction choice. Actual semantic-provider calls were not tested.
- The first memory is added without an LLM management call. Later updates consider top-k memories above the threshold. Historical UPDATE can revise analysis and reflection, but cannot overwrite the factual old vector/action/outcome with a different trade's data. ADD uses the actual new episode. DELETE and NONE are supported.
- Timestamps are taken from market observations, not wall-clock runtime, and serve as unique episode identifiers within a run. Temporary IDs are used only in management prompts.
- Prompt examples referring to timestamp-based mutation are superseded by the adapter's explicit ID contract. Invalid management output raises; it is not silently converted to an ADD.
- Horizon is three candles by default and is reflected in the decision prompt. Entry is the final observed close. CLI decisions advance by the horizon, so reflected outcomes cannot leak into earlier decisions.
- Historical memory databases are not reused by the CLI. Resume, shared multi-asset stores, concurrent writers, live streaming and exchange execution are outside this reconstruction.
- Original fixed chart names are isolated in per-step output directories. Concurrent graph calls within the same working directory are unsupported.

## Verification limits

Offline verification uses real libraries, tools, graph nodes, vectors and persistence, with injected deterministic LLM responses. It establishes software integration, not quality of real LLM reasoning or reproduction of reported experimental metrics. Invalid provider outputs stop the current run. A partial output directory is retained for diagnosis; reruns require a new output directory.

## Benchmark-folder extension

`run_graph.py` now accepts `--benchmark-dir` and `--benchmark-root/--assets` while retaining `--csv`. `benchmark_runner.py` implements one decision per sample, with all but the final horizon candles observed. It validates each selected folder, sorts by decision time, and defers the memory node until each outcome is observable. This replaces the chronological non-overlapping sliding schedule only for folder mode. All predictions still pass through the same technical agents and decision/reflection code.

Per-asset stores are isolated. Sample filenames disambiguate memories sharing the same decision timestamp; the original market decision timestamp is retained in metadata. Final outcomes are consolidated after the last prediction. Additional output files preserve sample ordering and the timing of memory changes. CSV column aliases and numeric Unix seconds/milliseconds are supported.
