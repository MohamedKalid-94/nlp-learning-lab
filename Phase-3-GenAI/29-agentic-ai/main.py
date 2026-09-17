from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    attempts: int
    found_answer: bool

def search_node(state: AgentState) -> AgentState:
    """Pretend this searches for an answer — fails on purpose for the first 2 attempts"""
    state["attempts"] += 1
    print(f"  Attempt {state['attempts']}: searching...")
    # Simulate success only on the 3rd attempt
    state["found_answer"] = state["attempts"] >= 3
    return state

def should_retry(state: AgentState) -> str:
    """
    This is the CONDITIONAL logic — the core of what makes this agentic
    rather than a fixed pipeline. Decides: loop back, or finish?
    """
    if state["found_answer"]:
        return "done"
    if state["attempts"] >= 5:  # safety cap, avoid infinite loops
        return "give_up"
    return "retry"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.set_entry_point("search")

# Conditional edge: after "search", check should_retry() to decide what's next
graph.add_conditional_edges(
    "search",
    should_retry,
    {
        "retry": "search",   # loop back to the SAME node
        "done": END,
        "give_up": END,
    }
)

app = graph.compile()
result = app.invoke({"query": "test", "attempts": 0, "found_answer": False})

print(f"\nFinal state: {result}")