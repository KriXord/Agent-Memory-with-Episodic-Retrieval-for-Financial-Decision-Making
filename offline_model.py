"""Deterministic test double, not an actual trading model or performance baseline."""
import json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

class OfflineModel:
    def bind_tools(self, tools, **kwargs):
        def call(messages):
            if hasattr(messages, "to_messages"):
                messages = messages.to_messages()
            tool = tools[0]
            return AIMessage(content="", tool_calls=[{
                "name": tool.name, "args": {}, "id": "offline-tool-call", "type": "tool_call"}])
        return RunnableLambda(call)

    def invoke(self, prompt):
        if isinstance(prompt, str):
            if "CURRENT MEMORY:" in prompt:
                return AIMessage(content='{"memory":[{"id":"NEW","event":"ADD"}]}')
            if "REQUIRED OUTPUT FORMAT (JSON)" in prompt:
                return AIMessage(content=json.dumps({
                    "decision": "LONG", "risk_reward_ratio": 1.5,
                    "confidence_level": "LOW", "primary_drivers": ["Offline fixture"],
                    "memory_insights": "Test only", "justification": "Deterministic smoke test"}))
            return AIMessage(content="Offline reflection: test the outcome and memory persistence, not profitability.")
        return AIMessage(content="Offline analysis: synthetic fixture; no financial inference is performed.")
