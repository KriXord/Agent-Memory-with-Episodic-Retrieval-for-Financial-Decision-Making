"""Public orchestrator; injectable models enable offline end-to-end verification."""
from copy import deepcopy
from default_config import DEFAULT_CONFIG
from graph_setup import SetGraph
from graph_util import TechnicalTools

class TradingGraph:
    def __init__(self, config=None, agent_llm=None, graph_llm=None):
        self.config = deepcopy(DEFAULT_CONFIG)
        if config:
            self.config.update({k:v for k,v in config.items() if k != "memory_config"})
            self.config["memory_config"].update(config.get("memory_config", {}))
        if agent_llm is None or graph_llm is None:
            from langchain_openai import ChatOpenAI
            agent_llm = agent_llm or ChatOpenAI(model=self.config["agent_llm_model"],
                          temperature=self.config["agent_llm_temperature"], max_retries=3)
            graph_llm = graph_llm or ChatOpenAI(model=self.config["graph_llm_model"],
                          temperature=self.config["graph_llm_temperature"], max_retries=3)
        self.agent_llm, self.graph_llm = agent_llm, graph_llm
        self.toolkit = TechnicalTools()
        self.graph = SetGraph(agent_llm, graph_llm, self.toolkit).set_graph()

    def invoke(self, state):
        state = {"messages": [], "horizon": self.config["horizon"],
                 "retrieval_mode": self.config["retrieval_mode"],
                 "memory_config": deepcopy(self.config["memory_config"]), **state}
        return self.graph.invoke(state)

    def refresh_llms(self):
        self.__init__(self.config)
