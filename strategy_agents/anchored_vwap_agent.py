"""Reconstructed Anchored VWAP agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_anchored_vwap_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'anchored_vwap', 'Anchored VWAP', 'Assess price relative to anchored VWAP, slope, tests and reclaims, breakouts, and distance from the volume-weighted benchmark. Respect the anchor shown; do not assume a daily session reset.')
