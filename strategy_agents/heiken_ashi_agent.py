"""Reconstructed Heiken Ashi agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_heiken_ashi_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'heiken_ashi', 'Heiken Ashi', 'Analyze candle color sequences, body size, upper/lower shadows, indecision, and trend transitions. Distinguish smoothed Heiken Ashi prices from executable market prices.')
