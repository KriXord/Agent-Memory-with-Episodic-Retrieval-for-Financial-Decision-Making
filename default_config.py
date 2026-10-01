DEFAULT_CONFIG = {
    "agent_llm_model": "gpt-4o-mini",
    "graph_llm_model": "gpt-4o",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "horizon": 3,
    "retrieval_mode": "indicator",
    "memory_config": {
        "top_k": 5, "similarity_threshold": 0.7, "max_memories": 10000,
        "normalize_vectors": False,
        "embedding_model": "text-embedding-3-small",
    },
}
