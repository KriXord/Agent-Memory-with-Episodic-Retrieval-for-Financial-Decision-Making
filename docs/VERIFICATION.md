# Verification

Verified with Python 3.12.14 and the direct dependencies pinned in requirements.txt.

Command: `python -m unittest discover -s tests -v`

Result: **17 tests passed**. The integration test ran two full LangGraph iterations, generated all ten active strategy charts per iteration, produced structured decisions and reflections using deterministic model doubles, retrieved the first memory on the second step, and persisted two memories. The standalone VWAP chart agent was exercised separately.

Additional checks: all Python files compile; a rendered RSI chart was visually inspected. Semantic routing was checked with an injected embedding provider. No real model/embedding API calls, exchange orders, or published experimental-result replication were performed.

The offline LONG outputs are predetermined test fixtures. They must not be presented as model performance.


Benchmark extension: verified one prediction per 100-row file (97 observed + 3 withheld), timestamp ordering despite shuffled filenames, delayed consolidation for overlapping horizons, duplicate decision-time identifiers, asset memory isolation, CSV aliases, Unix milliseconds, and preflight rejection of short files. The overlap timing test uses threshold -1 to exercise every eligible nonzero memory independent of similarity strength; production default remains 0.7. A separate offline CLI invocation with `--benchmark-dir` completed successfully at default settings. The user's actual benchmark CSVs were not available for testing.
