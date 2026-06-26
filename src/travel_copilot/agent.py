# agent.py
# ---------------------------------------------------------------------------
# A LangGraph agent that decides which tool to use to answer travel questions.
#
# THE LOOP:
#   agent node (LLM decides) --conditional edge--> tool node (runs tool)
#        ^                                              |
#        |__________________ loops back ________________|
#   ...until the agent gives a final answer --> END
# ---------------------------------------------------------------------------

from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode   # prebuilt node that executes tool calls

from . import config
from .tools import search_travel_guides, calculate


# --- STATE ------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- MODEL WITH TOOLS BOUND -------------------------------------------------
_tools = [search_travel_guides, calculate]
_model = ChatOllama(model=config.CHAT_MODEL, temperature=0, base_url=config.OLLAMA_BASE_URL)
_model_with_tools = _model.bind_tools(_tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a travel assistant. Use the search_travel_guides tool to find "
    "information about destinations, and the calculate tool for any arithmetic. "
    "Always base travel facts on the search tool's results, and cite the place "
    "and section. If you don't have enough information, say so."
))


# --- AGENT NODE (the brain: decide answer or tool) --------------------------
def agent_node(state: AgentState) -> dict:
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = _model_with_tools.invoke(messages)
    return {"messages": [response]}


# --- ROUTER for the conditional edge ----------------------------------------
def should_continue(state: AgentState) -> str:
    """Look at the agent's latest message and decide where to go next.

    If the agent requested a tool (the AI message has tool_calls), route to the
    tool node. Otherwise it gave a final answer, so end. THIS function is the
    'decision-making' that makes it an agent rather than a fixed pipeline.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"   # agent wants a tool -> go run it
    return END           # agent gave a final answer -> stop


# --- BUILD THE GRAPH --------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    # Two real nodes: the brain, and the hands (prebuilt ToolNode runs our tools).
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(_tools))

    # Flow:
    graph.add_edge(START, "agent")          # start at the agent

    # Conditional edge: after the agent, route based on should_continue().
    # The mapping says: if it returns "tools" -> tools node; if END -> finish.
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    # Normal edge: after running a tool, loop BACK to the agent so it can read
    # the result and decide again (answer, or another tool).
    graph.add_edge("tools", "agent")

    return graph.compile()


def ask_agent(question: str) -> str:
    """Run the agent on a question and return its final answer text."""
    app = build_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # Run with: poetry run python -m src.travel_copilot.agent
    # Try a few question types to see the agent choose different tools.
    for q in [
        "What is 15 times 4 plus 100?",                  # -> calculate tool
        "What is there to see in Aachen?",               # -> search tool
        "If a hotel in Aachen is 80 euros a night, how much for 4 nights?",  # -> both
        "What's the total for 3 nights at a hotel you find in the Aachen guide?", # -> both
    ]:
        print("=" * 70)
        print(f"Q: {q}\n")
        answer = ask_agent(q)
        print(f"A: {answer}")