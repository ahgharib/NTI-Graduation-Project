from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query: str) -> str:
    """
    Searches the web for educational resources and roadmap steps.
    Using DuckDuckGo (Free, no API key required).
    """
    search = DuckDuckGoSearchRun()
    try:
        # We limit the results to avoid token overflow
        results = search.invoke(f"how to learn {query} roadmap steps resources")
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

# Wrapper for the graph node
def search_node_func(state):
    query = state["user_request"]
    # If we are in a refinement loop, we might want to search based on feedback
    if state.get("feedback"):
        query = f"{query} fix: {state['feedback']}"
        
    results = search_web(query)
    return {"search_context": results}