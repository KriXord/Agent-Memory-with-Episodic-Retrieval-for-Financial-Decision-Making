"""Reconstructed Bollinger Bands agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_bollinger_bands_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'bollinger_bands', 'Bollinger Bands', 'Assess position relative to the middle and outer bands, bandwidth contraction/expansion, squeezes, breakouts, and mean reversion versus persistent band riding.')
