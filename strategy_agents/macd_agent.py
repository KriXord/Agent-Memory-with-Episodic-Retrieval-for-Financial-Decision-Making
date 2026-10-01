"""Reconstructed MACD agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_macd_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'macd', 'MACD', 'Analyze MACD/signal crossovers, the zero line, histogram sign and changes, momentum acceleration/deceleration, and possible price divergence.')
