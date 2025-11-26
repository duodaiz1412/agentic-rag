from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH, GREETING
from graph.chains import hallucination_grader, answer_grader, question_router
from graph.nodes import generate, grade_documents, retrieve, web_search, greeting, _is_greeting


load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["use_web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, GO TO WEB---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    documents_text = "\n\n-----\n\n".join([doc.page_content for doc in documents])
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents_text, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---CHECK ANSWER---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: ANSWER ADDRESSES THE USER QUESTION---")
            return "useful"
        else:
            print("---DECISION: ANSWER DOES NOT ADDRESS THE USER QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not_supported"


def route_question(state: GraphState):
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Check greeting/chit-chat FIRST (avoid unnecessary resource consumption)
    if _is_greeting(question):
        print("---DECISION: ROUTE QUESTION TO GREETING (CHIT-CHAT)---")
        return GREETING

    try:
        source = question_router.invoke({"question": question})
    except Exception as e:
        print(f"---ERROR: ROUTER FAILED ({type(e).__name__}: {e}), DEFAULTING TO RAG---")
        return RETRIEVE

    # Check if source is None, default to vectorstore
    if source is None:
        print("---WARNING: ROUTER RETURNED NONE, DEFAULTING TO RAG---")
        return RETRIEVE

    if source.datasource == WEBSEARCH:
        print("---DECISION: ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---DECISION: ROUTE QUESTION TO RAG---")
        return RETRIEVE
    
    # Fallback: if datasource is not web_search or vectorstore
    print("---WARNING: UNKNOWN DATASOURCE, DEFAULTING TO RAG---")
    return RETRIEVE


flow = StateGraph(state_schema=GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)
flow.add_node(GREETING, greeting)

flow.set_conditional_entry_point(
    route_question, path_map={RETRIEVE: RETRIEVE, WEBSEARCH: WEBSEARCH, GREETING: GREETING}
)

flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
flow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={"useful": END, "not_useful": WEBSEARCH, "not_supported": GENERATE},
)

flow.add_edge(WEBSEARCH, GENERATE)
flow.add_edge(GENERATE, END)
flow.add_edge(GREETING, END)  # Greeting node ends immediately, no need to continue

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
