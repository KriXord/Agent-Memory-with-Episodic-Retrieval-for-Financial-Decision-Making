"""Reconstructed RSI agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_rsi_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'rsi', 'RSI', 'Assess RSI relative to 30, 50 and 70, momentum persistence, exits from extremes, and possible price divergence. Overbought alone is not a sell signal.')
