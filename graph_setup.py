"""Sequential LangGraph retaining the uploaded expanded strategy-agent ordering."""
from importlib import import_module
from langgraph.graph import END, START, StateGraph
from agent_state import IndicatorAgentState
from decision_agent import create_final_trade_decider
from memory_agent import create_memory_agent
from strategy_agents.indicator_agent import create_indicator_agent

CHART_AGENTS = ("pattern", "trend", "bollinger_bands", "anchored_vwap", "sma",
                "stochastic", "fibonacci", "heiken_ashi", "rsi", "macd")

class SetGraph:
    def __init__(self, agent_llm, graph_llm, toolkit, tool_nodes=None):
        self.agent_llm, self.graph_llm, self.toolkit = agent_llm, graph_llm, toolkit

    def set_graph(self):
        graph = StateGraph(IndicatorAgentState)
        graph.add_node("indicator", create_indicator_agent(self.graph_llm, self.toolkit))
        graph.add_edge(START, "indicator")
        previous = "indicator"
        for name in CHART_AGENTS:
            module = import_module(f"strategy_agents.{name}_agent")
            creator = getattr(module, f"create_{name}_agent")
            graph.add_node(name, creator(self.agent_llm, self.graph_llm, self.toolkit))
            graph.add_edge(previous, name)
            previous = name
        graph.add_node("decision", create_final_trade_decider(self.graph_llm))
        graph.add_node("memory", create_memory_agent(self.graph_llm))
        graph.add_edge(previous, "decision")
        graph.add_edge("decision", "memory")
        graph.add_edge("memory", END)
        return graph.compile()
