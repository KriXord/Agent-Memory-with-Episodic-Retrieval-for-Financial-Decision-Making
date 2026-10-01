"""Reconstructed VWAP agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_vwap_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'vwap', 'VWAP', 'Assess support/resistance at VWAP, deviations and mean reversion, breaks/reclaims, slope and directional bias, and premium/discount to the volume-weighted benchmark. VWAP accumulates over the provided data; do not assume it resets daily.')
