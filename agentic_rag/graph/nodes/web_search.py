from typing import Any, Dict

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState


web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Search the web for documents.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the retrieved documents and the question
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] or []  # only relevant documents

    tavily_results = web_search_tool.invoke({"query": question})

    # get one huge string with all the results
    tavily_results_joined = "\n".join([res["content"] for res in tavily_results])

    # create a document object
    web_search_result = Document(
        page_content=tavily_results_joined,
        metadata={
            "doc_type": "web_search",
            "source": "tavily",
            "requires_enrollment": False,
        },
    )

    # append web search to the list of documents
    documents.append(web_search_result)

    return {
        "documents": documents,
        "question": question,
        "user_id": state.get("user_id"),
        "chat_history": state.get("chat_history", []),
    }
