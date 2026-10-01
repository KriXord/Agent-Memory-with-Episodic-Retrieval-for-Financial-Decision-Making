"""Post-trade consolidation node reconstructed from supplied store and graph interfaces."""
import numpy as np
from memory import TradingMemory, TradingMemorySystem, analysis_text
from graph_util import TechnicalTools


def create_memory_agent(llm):
    def memory_agent_node(state):
        if state.get("defer_memory_update", False):
            return {}
        data = state["kline_data"]
        vector = np.concatenate([TechnicalTools.compute_all_trading_indicators(data),
                                 TechnicalTools.compute_normalized_ohlcv_vector(data)])
        timestamp = data["Datetime"][-1]
        horizon = state.get("horizon", 3)
        available_at = state["orig_kline_data"]["Datetime"][len(data["Close"]) - 1 + horizon]
        current = TradingMemory(
            timestamp=timestamp + ("#" + state["sample_id"] if state.get("sample_id") else ""), market_vector=vector,
            analysis_text=analysis_text(state["report_dict"]),
            trade_action=state["final_trade_decision"], trade_outcome=state["trade_outcome"],
            reflection=state["reflection"], metadata={
                "symbol": state["stock_name"], "time_frame": state["time_frame"],
                "available_at": available_at, "decision_analysis": state["analysis_results"],
                "horizon": horizon, "decision_timestamp": timestamp, "sample_id": state.get("sample_id")})
        store = TradingMemorySystem(llm, state["memory_file"], **state.get("memory_config", {}))
        counts = store.update_memory_with_llm(current, state.get("retrieval_mode", "indicator"))
        return {"memory_actions": counts, "memory_count": len(store.memories)}
    return memory_agent_node
