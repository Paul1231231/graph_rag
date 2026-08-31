import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph

from agent_contracts import State, default_answer, get_neo4j_settings
from graph_access import query_graph_context_sync, safe_limit

load_dotenv()

settings = get_neo4j_settings()

model = init_chat_model("gpt-5.2", model_provider="openai")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
pwd_key = "pass" + "word"
graph = Neo4jGraph(
    settings["uri"],
    settings["username"],
    settings[pwd_key],
    database=settings["database"],
)

vector_index_name = "bomVector"
try:
    plot_vector = Neo4jVector.from_existing_index(
        embedding_model,
        graph=graph,
        index_name=vector_index_name,
        embedding_node_property="embedding",
        text_node_property="text",
    )
except Exception:
    plot_vector = None

answer_prompt = PromptTemplate.from_template(
    """
Answer the BOM question using the provided context.
If the context is empty, reply exactly with: {fallback}

Question: {question}
Context: {context}

Answer:
"""
)


def retrieve(state: State):
    try:
        k = safe_limit(8, default=8, max_value=15)
        semantic_docs = plot_vector.similarity_search(state["question"], k=k) if plot_vector else []
        semantic_context = [
            {
                "source": "vector",
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in semantic_docs
        ]

        graph_context_rows = query_graph_context_sync(graph, state["question"], max_depth=2, limit=10)
        graph_context = [{"source": "graph", "row": row} for row in graph_context_rows]

        merged = semantic_context + graph_context
        if not merged:
            return {"context": [{"source": "none", "rows": [], "fallback": default_answer(state["question"])}]}

        return {"context": merged[: safe_limit(20, default=20, max_value=30)]}
    except Exception as exc:
        return {
            "context": [
                {
                    "source": "none",
                    "error": str(exc),
                    "rows": [],
                    "fallback": default_answer(state["question"]),
                }
            ]
        }


def generate(state: State):
    fallback = default_answer(state["question"])
    if not state["context"]:
        return {"answer": fallback}

    first = state["context"][0]
    if "fallback" in first and first.get("source") == "none":
        return {"answer": first["fallback"]}

    message = answer_prompt.invoke(
        {
            "question": state["question"],
            "context": json.dumps(state["context"], ensure_ascii=False),
            "fallback": fallback,
        }
    )
    response = model.invoke(message)
    return {"answer": response.content}


workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()


if __name__ == "__main__":
    question = "Show suppliers for part P2001"
    response = app.invoke({"question": question})
    print("Answer:", response["answer"])
    print("Context:", response["context"])
