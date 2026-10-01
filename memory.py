"""Reconstructed episodic store. Single writer; JSONL persistence is explicit and atomic."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import copy
import json
import os
import tempfile
import numpy as np

@dataclass
class TradingMemory:
    timestamp: str
    market_vector: np.ndarray
    analysis_text: str
    analysis_image: Optional[list] = None
    trade_action: Optional[str] = None
    trade_outcome: Optional[float] = None
    metadata: Optional[dict] = None
    reflection: Optional[str] = None
    semantic_vector: Optional[list] = None

    def to_dict(self):
        data = asdict(self)
        data["market_vector"] = np.asarray(self.market_vector, dtype=float).tolist()
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["market_vector"] = np.asarray(data["market_vector"], dtype=float)
        return cls(**data)


def analysis_text(reports):
    if isinstance(reports, str):
        return reports
    return "\n\n".join(f"{key}: {value}" for key, value in sorted(reports.items()))


class TradingMemorySystem:
    def __init__(self, llm=None, memory_file="trading_memories.jsonl", max_memories=10000,
                 similarity_threshold=0.7, top_k=5, normalize_vectors=False,
                 embedding_model="text-embedding-3-small", embeddings=None):
        if max_memories < 1 or top_k < 1 or not -1 <= similarity_threshold <= 1:
            raise ValueError("Invalid memory capacity, top_k, or threshold")
        self.llm, self.memory_file = llm, Path(memory_file)
        self.max_memories, self.top_k = max_memories, top_k
        self.similarity_threshold = similarity_threshold
        self.normalize_vectors = normalize_vectors
        self.embedding_model, self.embeddings = embedding_model, embeddings
        self.memories = []
        self.load_memories()

    @staticmethod
    def remove_code_blocks(text):
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return text

    @staticmethod
    def _vector(value):
        vector = np.asarray(value, dtype=float).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("Vectors must be nonempty and finite")
        return vector

    def load_memories(self):
        if not self.memory_file.exists():
            return
        loaded = []
        for line in self.memory_file.read_text().splitlines():
            if line.strip():
                memory = TradingMemory.from_dict(json.loads(line))
                memory.market_vector = self._vector(memory.market_vector)
                loaded.append(memory)
        if len({m.timestamp for m in loaded}) != len(loaded):
            raise ValueError("Duplicate memory timestamps")
        if len({m.market_vector.size for m in loaded}) > 1:
            raise ValueError("Mixed vector dimensions in memory file")
        self.memories = loaded

    def save_memories(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.memory_file.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                for memory in self.memories:
                    stream.write(json.dumps(memory.to_dict(), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.memory_file)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def add_memory(self, market_vector, analysis_text, analysis_image=None, trade_action=None,
                   trade_outcome=None, metadata=None, reflection=None, timestamp=None):
        vector = self._vector(market_vector)
        if self.memories and vector.size != self.memories[0].market_vector.size:
            raise ValueError("Memory vector dimensions do not match")
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        if self.find_memory_by_timestamp(timestamp):
            raise ValueError("Memory timestamp already exists")
        memory = TradingMemory(timestamp, vector, analysis_text, analysis_image,
                               trade_action, trade_outcome, metadata or {}, reflection)
        self.memories.append(memory)
        self.memories = self.memories[-self.max_memories:]
        return memory

    def find_memory_by_timestamp(self, timestamp):
        return next((m for m in self.memories if m.timestamp == timestamp), None)

    def delete_memory(self, timestamp):
        target = self.find_memory_by_timestamp(timestamp)
        if target is None:
            return False
        self.memories = [m for m in self.memories if m.timestamp != timestamp]
        return True

    def _embed(self, text):
        if self.embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        return self._vector(self.embeddings.embed_query(text))

    def retrieve_similar_memories(self, query_vector, report_dict=None, mode="indicator",
                                  top_k=None, min_similarity=None):
        if mode not in {"indicator", "semantic"}:
            raise ValueError("mode must be indicator or semantic")
        k = self.top_k if top_k is None else top_k
        threshold = self.similarity_threshold if min_similarity is None else min_similarity
        if k < 1 or not -1 <= threshold <= 1:
            raise ValueError("Invalid retrieval settings")
        if not self.memories:
            return []
        if mode == "indicator":
            q = self._vector(query_vector)
            matrix = np.vstack([m.market_vector for m in self.memories])
        else:
            if report_dict is None:
                raise ValueError("Semantic retrieval needs the current analysis reports")
            q = self._embed(analysis_text(report_dict))
            vectors = []
            for memory in self.memories:
                # Cache belongs to this embedding model; do not reuse another model's vectors.
                if memory.semantic_vector is None or (memory.metadata or {}).get("embedding_model") != self.embedding_model:
                    memory.semantic_vector = self._embed(memory.analysis_text).tolist()
                    memory.metadata = {**(memory.metadata or {}), "embedding_model": self.embedding_model}
                vectors.append(memory.semantic_vector)
            matrix = np.asarray(vectors, dtype=float)
        if matrix.shape[1] != q.size:
            raise ValueError("Query and memory vector dimensions do not match")
        if self.normalize_vectors and mode == "indicator" and len(matrix) >= 2:
            mean, std = matrix.mean(axis=0), matrix.std(axis=0)
            std[std < 1e-12] = 1.0
            matrix, q = (matrix - mean) / std, (q - mean) / std
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        scores = np.divide(matrix @ q, norms, out=np.zeros(len(matrix)), where=norms > 1e-12)
        order = np.argsort(-scores, kind="stable")
        return [(self.memories[i], float(scores[i])) for i in order if norms[i] > 1e-12 and scores[i] >= threshold][:k]

    def build_update_prompt(self, current_memory, mode="indicator"):
        from memory_prompts import get_update_trading_memory_messages
        similar = self.retrieve_similar_memories(current_memory.market_vector,
                    current_memory.analysis_text, mode=mode)
        mapping = {str(i): m.timestamp for i, (m, _) in enumerate(similar)}
        history = []
        for i, (memory, score) in enumerate(similar):
            row = memory.to_dict()
            for key in ("market_vector", "semantic_vector", "analysis_image"):
                row.pop(key, None)
            row.update(id=str(i), similarity_score=score)
            history.append(row)
        current = current_memory.to_dict()
        for key in ("market_vector", "semantic_vector", "analysis_image"):
            current.pop(key, None)
        current["id"] = "NEW"
        return get_update_trading_memory_messages(history, current), mapping

    def apply_memory_updates(self, llm_response, id_to_timestamp, current_memory):
        payload = json.loads(self.remove_code_blocks(llm_response))
        items = payload.get("memory")
        if not isinstance(items, list) or not items:
            raise ValueError("Expected a nonempty memory event list")
        # Validate the full transaction before applying any operations.
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Memory events must be objects")
            ident, event = str(item.get("id", "")), item.get("event")
            if ident in seen:
                raise ValueError("Duplicate memory event ID")
            seen.add(ident)
            if ident == "NEW":
                if event not in {"ADD", "NONE"}:
                    raise ValueError("NEW allows only ADD/NONE")
            elif ident not in id_to_timestamp or event not in {"UPDATE", "DELETE", "NONE"}:
                raise ValueError("Unknown memory ID or event")
            if event == "UPDATE":
                for key in ("analysis_text", "reflection"):
                    if key in item and not isinstance(item[key], str):
                        raise ValueError("Updated analysis/reflection must be text")
        if "NEW" not in seen:
            raise ValueError("Memory manager must decide on NEW")
        before = copy.deepcopy(self.memories)
        counts = dict.fromkeys(("ADD", "UPDATE", "DELETE", "NONE"), 0)
        try:
            # Apply historical changes first, before capacity pruning on ADD.
            for item in sorted(items, key=lambda x: str(x["id"]) == "NEW"):
                ident, event = str(item["id"]), item["event"]
                if event == "ADD":
                    data = current_memory.to_dict()
                    data.pop("semantic_vector")
                    self.add_memory(**data)
                elif event == "UPDATE":
                    target = self.find_memory_by_timestamp(id_to_timestamp[ident])
                    if target is None:
                        raise ValueError("Target memory disappeared")
                    # Preserve the factual state/action/outcome of the historical episode.
                    for key in ("analysis_text", "reflection"):
                        if key in item:
                            setattr(target, key, item[key])
                    target.semantic_vector = None
                elif event == "DELETE":
                    self.delete_memory(id_to_timestamp[ident])
                counts[event] += 1
            self.save_memories()
        except Exception:
            self.memories = before
            raise
        return counts

    def update_memory_with_llm(self, current_memory, mode="indicator"):
        if not self.memories:
            return self.apply_memory_updates('{"memory":[{"id":"NEW","event":"ADD"}]}', {}, current_memory)
        if self.llm is None:
            raise ValueError("Memory management requires an LLM")
        prompt, mapping = self.build_update_prompt(current_memory, mode)
        return self.apply_memory_updates(self.llm.invoke(prompt).content, mapping, current_memory)

    def get_statistics(self):
        return {"total_memories": len(self.memories)}
