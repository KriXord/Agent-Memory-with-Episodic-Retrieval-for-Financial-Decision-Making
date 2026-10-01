from memory import TradingMemory, TradingMemorySystem
from graph_util import TechnicalTools
import json
import numpy as np

"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports with historical trade memories to issue a LONG or SHORT order.
"""

def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    along with similar past trade memories and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """
    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        macd_report = state["macd_report"]
        anchored_vwap_report = state["anchored_vwap_report"]
        heiken_ashi_report = state["heiken_ashi_report"]
        sma_report = state["sma_report"]
        stochastic_report = state["stochastic_report"]
        rsi_report = state["rsi_report"]
        bollinger_bands_report = state["bollinger_bands_report"]
        fibonacci_report = state["fibonacci_report"]

        time_frame = state["time_frame"]
        horizon = state.get("horizon", 3)
        stock_name = state["stock_name"]

        orig_kline_data = state["orig_kline_data"]

        kline_data = state["kline_data"]

        memory_file = state["memory_file"]

        memory_system = TradingMemorySystem(llm, memory_file, **state.get("memory_config", {}))

        indicator_vec = TechnicalTools.compute_all_trading_indicators(kline_data)
        ohlcv_vec = TechnicalTools.compute_normalized_ohlcv_vector(kline_data)

        market_vec = np.concatenate([indicator_vec, ohlcv_vec])

        report_dict = {
            "indicator": indicator_report,
            "pattern": pattern_report,
            "trend": trend_report,
            "macd": macd_report,
            "anchored_vwap": anchored_vwap_report,
            "heiken_ashi": heiken_ashi_report,
            "sma": sma_report,
            "stochastic": stochastic_report,
            "rsi": rsi_report,
            "bollinger_bands": bollinger_bands_report,
            "fibonacci": fibonacci_report,
        }

        similar_memories = memory_system.retrieve_similar_memories(market_vec, report_dict, mode=state.get("retrieval_mode", "indicator"))
        # similar_memories = []

        # --- Build single big prompt string ---
        prompt = f"""You are an elite HFT quantitative analyst making real-time trade decisions on {stock_name} using {time_frame} timeframe data. 

**MISSION:** Issue an immediate execution order (LONG or SHORT only - HOLD is prohibited) based on comprehensive analysis.

**FORECAST HORIZON:** Predict market direction for the next {horizon} candlesticks ({time_frame} period).

---

**HISTORICAL CONTEXT - Similar Market Conditions:**
"""

        if not similar_memories:
            prompt += "No highly relevant past trade memories found for current market conditions.\n"
        else:
            for i, (memory, score) in enumerate(similar_memories, start=1):
                prompt += (
                    f"\n**Memory {i}** (Similarity: {score:.2f})\n"
                    f"• Date: {memory.timestamp}\n"
                    f"• Analysis: {memory.analysis_text}\n"
                    f"• Decision: {memory.trade_action}\n"
                    f"• Outcome: {memory.trade_outcome}\n"
                    f"• Reflection: {memory.reflection}\n"
                    f"• Context: {memory.metadata if memory.metadata else 'N/A'}\n"
                )


        prompt += f"""

---

**CURRENT MARKET ANALYSIS:**

**Technical Indicators:**
{indicator_report}

**Pattern Analysis:**
{pattern_report}

**Trend Analysis:**
{trend_report}

**MACD Analysis:**
{macd_report}

**Anchored VWAP Analysis*
{anchored_vwap_report}

**Heiken Ashi Analysis**
{heiken_ashi_report}

**SMA Analysis**
{sma_report}

**Stochastic Analysis**
{stochastic_report}

**Bollinger Bands Analysis**
{bollinger_bands_report}

**Fibonacci Analysis**
{fibonacci_report}

**RSI Analysis**
{rsi_report}

---

**DECISION FRAMEWORK:**

**Primary Signals (High Weight):**
• Strong momentum confirmations (MACD crossovers, RSI breakouts >70 or <30)
• Completed breakout patterns with volume confirmation
• Clear trend line breaks with decisive price action

**Secondary Signals (Medium Weight):**
• Multiple indicator alignment in same direction
• Support/resistance level interactions
• Pattern formations nearing completion

**Risk Filters:**
• Avoid conflicting signals unless one direction shows overwhelming strength
• Consider historical outcomes from similar setups in your memory analysis
• Factor in current volatility for risk-reward calculations
• Suggest a reasonable **risk-reward ratio** between **1.2 and 1.8**, based on current volatility and prediction strength.

**Memory Integration:**
• Reference similar past scenarios and their outcomes
• Learn from successful/failed trades in comparable market conditions
• Adjust confidence based on historical performance patterns

---

**REQUIRED OUTPUT FORMAT (JSON):**
**Return only valid JSON in this format:**

```json
{{
    "forecast_horizon": "Next {time_frame} candlestick prediction",
    "decision": "LONG or SHORT",
    "confidence_level": "HIGH/MEDIUM/LOW",
    "primary_drivers": ["List 2-3 strongest signals supporting decision"],
    "memory_insights": "Key lessons from similar historical scenarios",
    "justification": "Concise reasoning combining current analysis with historical context",
    "risk_reward_ratio": "<float between 1.2 and 1.8>",
    "stop_loss_rationale": "Brief explanation of risk management approach"
}}

**Decision Hierarchy:**

1. **STRONG ALIGNMENT:** All reports + positive memory outcomes → High confidence trade

2. **MODERATE ALIGNMENT:** 2/3 reports align + supportive memory → Medium confidence trade

3. **WEAK SIGNALS:** Mixed reports → Choose direction with strongest recent momentum + best historical precedent

Execute with precision. The market waits for no one.
"""

        # --- LLM call for decision ---
        response = llm.invoke(prompt)
        raw_text = response.content

        # print("\n[DEBUG] Decision Agent Response (Final decision analysis):")
        # print(raw_text)

        raw_text = TradingMemorySystem.remove_code_blocks(raw_text)

        parsed = json.loads(raw_text)
        final_trade_decision = str(parsed.get("decision", "")).upper()
        if final_trade_decision not in {"LONG", "SHORT"}:
            raise ValueError("Decision must be LONG or SHORT; invalid responses are not scored")
        risk_reward_ratio = float(parsed.get("risk_reward_ratio") or 1.5)
        if not np.isfinite(risk_reward_ratio) or not 1.2 <= risk_reward_ratio <= 1.8:
            raise ValueError("risk_reward_ratio must be between 1.2 and 1.8")

        # Combine everything else into one analysis string
        excluded_keys = {"decision"}
        analysis_parts = []

        for key, value in parsed.items():
            if key in excluded_keys:
                continue
            if key == "primary_drivers" and isinstance(value, list):
                # Convert list to a bullet-point string
                value_str = "\n  - " + "\n  - ".join(value)
            else:
                value_str = str(value)
            analysis_parts.append(f"{key}: {value_str}")

        analysis_results = "\n".join(analysis_parts)

        # print("\n[DEBUG] Decision Agent Final Trade Decision:")
        # print(final_trade_decision)

        # print("\n[DEBUG] Decision Agent Analysis result:")
        # print(analysis_results)

        # --- Simulate outcome ---
        # Assume decision is made at last candle in kline_data
        decision_index = len(kline_data["Close"]) - 1# if you're predicting at 90 of 100
        trade_outcome = TechnicalTools.simulate_trade_outcome(orig_kline_data, decision_index, final_trade_decision, horizon=horizon)


        if trade_outcome["outcome"] == "INVALID":
            raise ValueError("Full evaluation horizon is unavailable")

        reflection_prompt = f"""You are conducting a post-trade analysis as an elite HFT quantitative analyst. 

        **TRADE SUMMARY:**
        • Stock: {stock_name}
        • Timeframe: {time_frame}
        • Decision Made: {final_trade_decision}
        • Confidence Level: {parsed.get('confidence_level', 'N/A')}
        • Expected Risk-Reward: {parsed.get('risk_reward_ratio', 'N/A')}

        **ORIGINAL ANALYSIS SUMMARY:**
        {analysis_results}

        **ACTUAL TRADE OUTCOME:**
        • P&L Result: {trade_outcome['pnl']:.4f}
        • Trade Status: {'PROFITABLE' if trade_outcome['pnl'] > 0 else 'LOSS' if trade_outcome['pnl'] < 0 else 'BREAKEVEN'}
        • Market Behavior: {trade_outcome.get('market_direction', 'N/A')}

        **REFLECTION FRAMEWORK:**

        **1. Decision Quality Assessment:**
        • Was the original analysis sound given available information?
        • Did the primary drivers correctly predict market movement?
        • How well did historical memory insights apply to this scenario?

        **2. Execution Analysis:**
        • Were the technical indicators correctly interpreted?
        • Did pattern analysis accurately forecast price action?
        • Was the confidence level appropriate for the outcome?

        **3. Learning Opportunities:**
        • What market conditions were underestimated or overestimated?
        • Which signals proved most/least reliable in this instance?
        • How could similar setups be approached differently in the future?

        **4. Memory Integration:**
        • How does this outcome update understanding of similar market conditions?
        • What new patterns or correlations emerged from this trade?
        • Should weight be adjusted for specific indicators based on this result?

        **REQUIRED OUTPUT:**
        Provide a concise but thorough reflection (3-5 sentences) covering:
        - Key factors that led to the outcome
        - What worked well or poorly in the analysis
        - Specific lessons learned for future similar scenarios
        - Any adjustments to analytical approach or confidence calibration

        Focus on actionable insights that will improve future trading decisions in similar market conditions.
        """

        reflection_response = llm.invoke(reflection_prompt)
        reflection_text = reflection_response.content.strip()


        return {
            "final_trade_decision": final_trade_decision,  # only LONG/SHORT
            "analysis_results": analysis_results,            # combined text
            "messages": [response],
            "decision_prompt": prompt,
            "trade_outcome": trade_outcome["pnl"],
            "reflection": reflection_text,
            "risk_reward_ratio": risk_reward_ratio,
            "retrieved_memory_count": len(similar_memories),
            "report_dict": report_dict,
            "trade_outcome_details": trade_outcome
        }

    return trade_decision_node

