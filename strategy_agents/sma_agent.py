"""Reconstructed SMA agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_sma_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'sma', 'SMA', 'Compare short and long moving averages, their slopes and crossovers. Assess price above/below the averages, support/resistance, trend alignment, and lag.')
