from typing import Annotated, Sequence, TypedDict, List
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class IndicatorAgentState(TypedDict):
    """State type for the Indicator Agent including messages, input data, and analysis result."""
    orig_kline_data: Annotated[dict, "Original full kline data across 100 periods"]
    kline_data: Annotated[dict, "OHLCV dictionary used for computing technical indicators"]
    time_frame: Annotated[str, "time period for k line data provided"]
    stock_name: Annotated[str, "stock name for prompt"]
    memory_file: Annotated[str, "jsonl file that saves the memory"]

    # Trading Memory System
    
    # Indicator Agent Tools output values (explicitly added per indicator)
    rsi: Annotated[List[float], "Relative Strength Index values"]
    macd: Annotated[List[float], "MACD line values"]
    macd_signal: Annotated[List[float], "MACD signal line values"]
    macd_hist: Annotated[List[float], "MACD histogram values"]
    stoch_k: Annotated[List[float], "Stochastic Oscillator %K values"]
    stoch_d: Annotated[List[float], "Stochastic Oscillator %D values"]
    roc: Annotated[List[float], "Rate of Change values"]
    willr: Annotated[List[float], "Williams %R values"]
    anchored_vwap: Annotated[List[float], "Anchored Volume-Weighted Average Price"]
    vwap: Annotated[List[float], "Volume-Weighted Average Price"]
    st_sma: Annotated[List[float], "Short term Simple Moving Average"]
    lt_sma: Annotated[List[float], "Long term Simple Moving Average"]
    heiken_ashi_data: Annotated[dict, "Dictionary used for the Heiken Ashi version of OHLC"]
    indicator_report: Annotated[str, "Final indicator agent summary report to be used by downstream agents"]


    # Pattern Agent
    pattern_image: Annotated[str, "Base64-encoded K-line chart for pattern agent use"]
    pattern_image_filename: Annotated[str, "Local file path to saved K-line chart image"]
    pattern_image_description: Annotated[str, "Brief description of the generated K-line image"]
    pattern_report: Annotated[str, "Final pattern agent summary report to be used by downstream agents"]

    # Trend Agent
    trend_image: Annotated[str, "Base64-encoded trend-annotated candlestick (K-line) chart for trend agent use"]
    trend_image_filename: Annotated[str, "Local file path to saved trendline-enhanced K-line chart image"]
    trend_image_description: Annotated[str, "Brief description of the chart, including presence of support/resistance lines and visual characteristics"]
    trend_report: Annotated[str, "Final trend analysis summary, describing structure, directional bias, and technical observations for downstream agents"]

    # Anchored VWAP Agent
    anchored_vwap_image: Annotated[str, "Base64-encoded Anchored VWAP K-line chart for anchored vwap agent use"]
    anchored_vwap_image_filename: Annotated[str, "Local file path to saved Anchored VWAP chart image"]
    anchored_vwap_image_description: Annotated[str, "Brief description of the anchored vwap image"]
    anchored_vwap_report: Annotated[str, "Anchored VWAP analysis summary report"]

    # VWAP Agent
    vwap_image: Annotated[str, "Base64-encoded VWAP K-line chart for vwap agent use"]
    vwap_image_filename: Annotated[str, "Local file path to saved VWAP chart image"]
    vwap_image_description: Annotated[str, "Brief description of the vwap image"]
    vwap_report: Annotated[str, "VWAP analysis summary report"]

    # Heiken Ashi Agent
    heiken_ashi_image: Annotated[str, "Base64-encoded Heiken Ashi K-line chart for Heiken Ashi agent use"]
    heiken_ashi_image_filename: Annotated[str, "Local file path to saved Heiken Ashi chart image"]
    heiken_ashi_image_description: Annotated[str, "Brief description of the heiken ashi image"]
    heiken_ashi_report: Annotated[str, "Heiken Ashi analysis summary report"]

    # MACD Agent
    macd_image: Annotated[str, "Base64-encoded MACD K-line chart for MACD agent use"]
    macd_image_filename: Annotated[str, "Local file path to saved MACD chart image"]
    macd_image_description: Annotated[str, "Brief description of the macd image"]
    macd_report: Annotated[str, "MACD analysis summary report"]

    # SMA Agent
    sma_image: Annotated[str, "Base64-encoded SMA chart showing short and long term moving averages for SMA agent use"]
    sma_image_filename: Annotated[str, "Local file path to saved SMA chart image"]
    sma_image_description: Annotated[str, "Brief description of the sma image"]
    sma_report: Annotated[str, "SMA analysis summary report"]

    # Stochastic Agent
    stochastic_image: Annotated[str, "Base64-encoded Stochastic Oscillator K-line chart for stochastic pattern agent use"]
    stochastic_image_filename: Annotated[str, "Local file path to saved Stochastic chart image"]
    stochastic_image_description: Annotated[str, "Brief description of the stochastic image"]
    stochastic_report: Annotated[str, "Stochastic analysis summary report"]

    # RSI Agent
    rsi_image: Annotated[str, "Base64-encoded RSI K-line chart for RSI agent use"]
    rsi_image_filename: Annotated[str, "Local file path to saved RSI chart image"]
    rsi_image_description: Annotated[str, "Brief description of the rsi image"]
    rsi_report: Annotated[str, "RSI analysis summary report"]

    # Bollinger Bands Agent
    bollinger_bands_image: Annotated[str, "Base64-encoded Bollinger Bands K-line chart for bollinger bands agent use"]
    bollinger_bands_image_filename: Annotated[str, "Local file path to saved Bollinger Bands chart image"]
    bollinger_bands_image_description: Annotated[str, "Brief description of the bollinger bands image"]
    bollinger_bands_report: Annotated[str, "Bollinger Bands analysis summary report"]

    # Fibonacci Agent
    fibonacci_image: Annotated[str, "Base64-encoded Fibonacci retracement K-line chart for fibonacci agent use"]
    fibonacci_image_filename: Annotated[str, "Local file path to saved Fibonacci chart image"]
    fibonacci_image_description: Annotated[str, "Brief description of the fibonacci image"]
    fibonacci_report: Annotated[str, "Fibonacci analysis summary report"]


    # Final analysis and messaging context
    analysis_results: Annotated[str, "Computed result of the analysis or decision"]
    messages: Annotated[List[BaseMessage], "List of chat messages used in LLM prompt construction"]
    decision_prompt: Annotated[str, "decision prompt for reflection"]
    final_trade_decision: Annotated[str, "Final BUY or SELL decision made after analyzing indicators"]
    trade_outcome: Annotated[float, "Final trade outcome base on final_trade_decision"]
    reflection: Annotated[str, "Reflection of the current trade trial base on outcome, summaring the current trade analysis, what are some advantages, and what needs to be improved"]
    risk_reward_ratio: Annotated[float, "Float between 1.2 and 1.8"]

    horizon: int
    retrieval_mode: str
    memory_config: dict
    report_dict: dict
    trade_outcome_details: dict
    retrieved_memory_count: int
    memory_actions: dict
    memory_count: int

    defer_memory_update: bool
    sample_id: str
