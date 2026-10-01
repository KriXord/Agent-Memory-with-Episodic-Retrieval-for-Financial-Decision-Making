"""Shared VWAP-style workflow: tool request -> chart -> vision analysis -> state."""
import copy
import json
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError, APIConnectionError


def invoke_with_retry(call_fn, *args, retries=3, wait_sec=1):
    for attempt in range(retries):
        try:
            return call_fn(*args)
        except (RateLimitError, APIConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(wait_sec * 2 ** attempt)


def create_chart_agent(tool_llm, graph_llm, toolkit, name, title, analysis_prompt):
    tool = getattr(toolkit, f"generate_{name}_image")
    def node(state):
        messages = [SystemMessage(content=(
            f"You are a {title} analysis assistant in a short-horizon trading context. "
            f"First call {tool.name} using the supplied kline_data. "
            "Do not predict before generating and analyzing the chart.")),
            HumanMessage(content="Recent kline data:\n" + json.dumps(state["kline_data"]))]
        chain = tool_llm.bind_tools([tool], tool_choice=tool.name)
        response = invoke_with_retry(chain.invoke, messages)
        messages.append(response)
        result = None
        for call in response.tool_calls:
            if call["name"] != tool.name:
                raise ValueError(f"Unexpected chart tool: {call['name']}")
            args = dict(call["args"])
            args["kline_data"] = copy.deepcopy(state["kline_data"])
            result = tool.invoke(args)
            messages.append(ToolMessage(tool_call_id=call["id"], content=json.dumps({
                "generated": bool(result.get(f"{name}_image")),
                "description": result.get(f"{name}_image_description", "")})))
        if result is None or not result.get(f"{name}_image"):
            raise RuntimeError(f"{title} chart was not generated")
        image = result[f"{name}_image"]
        image_prompt = [
            {"type": "text", "text": f"{title} chart, {state['time_frame']} candles.\n" + analysis_prompt
             + "\nProvide directional bias, concrete evidence, conflicting signals, and uncertainty. "
             "Only describe volume confirmation if volume is visible in the chart."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]
        final = invoke_with_retry(graph_llm.invoke, [
            SystemMessage(content=f"You specialize in {title} chart analysis."),
            HumanMessage(content=image_prompt)])
        return {"messages": messages + [final], f"{name}_report": final.content,
                f"{name}_image": image}
    return node
