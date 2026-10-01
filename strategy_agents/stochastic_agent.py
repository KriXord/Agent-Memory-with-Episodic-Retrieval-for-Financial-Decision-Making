"""Reconstructed Stochastic Oscillator agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_stochastic_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'stochastic', 'Stochastic Oscillator', 'Analyze %K/%D crossovers, 20/80 zones, exits from extremes, momentum shifts, and whether the trend supports or contradicts reversal signals.')
