"""Reconstructed Fibonacci Retracement agent following the supplied VWAP agent format."""
from .chart_agent import create_chart_agent


def create_fibonacci_agent(tool_llm, graph_llm, toolkit):
    return create_chart_agent(tool_llm, graph_llm, toolkit,
                              'fibonacci', 'Fibonacci Retracement', 'Analyze price interaction with 23.6%, 38.2%, 50%, 61.8%, 78.6% retracement levels. Discuss swing context, confluence, rejection, breaks, and invalidation.')
