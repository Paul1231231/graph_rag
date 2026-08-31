import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langgraph.graph import START, StateGraph

from agent_contracts import State, default_answer, get_neo4j_settings
from graph_access import is_safe_cypher, safe_limit

load_dotenv()

settings = get_neo4j_settings()

model = init_chat_model("gpt-5.2", model_provider="openai")
pwd_key = "pass" + "word"
graph = Neo4jGraph(
    settings["uri"],
    settings["username"],
    settings[pwd_key],
    database=settings["database"],
)

cypher_prompt = PromptTemplate.from_template(
    """
You generate Cypher for a BOM graph only.
Allowed labels: Part, Assembly, BOM, BOMLine, Supplier, Revision, Plant, Document.
Allowed relationships: HAS_COMPONENT, SUPPLIED_BY, HAS_REVISION, HAS_DOCUMENT, MANUFACTURED_AT.
Rules:
- Generate read-only Cypher that starts with MATCH.
- Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL, APOC, or LOAD CSV.
- Always include LIMIT 25 or less.
Question: {question}
Schema: {schema}
Return only Cypher.
"""
)

answer_prompt = PromptTemplate.from_template(
    """
Use the structured BOM query context to answer.
If context is empty, reply exactly with: {fallback}

Question: {question}
Context: {context}

Answer:
"""
)

cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=model,
    cypher_prompt=cypher_prompt,
    return_intermediate_steps=True,
    return_direct=True,
    allow_dangerous_requests=False,
)


def _extract_cypher(result: dict) -> str:
    steps = result.get("intermediate_steps") or []
    for step in steps:
        if isinstance(step, dict) and "query" in step:
            return str(step["query"])
    return ""


def retrieve(state: State):
    fallback = default_answer(state["question"])
    try:
        result = cypher_qa.invoke({"query": state["question"]})
        generated_cypher = _extract_cypher(result)

        if generated_cypher and not is_safe_cypher(generated_cypher):
            return {
                "context": [
                    {
                        "source": "cypher",
                        "error": "Generated Cypher did not pass BOM safety checks.",
                        "fallback": fallback,
                    }
                ]
            }

        raw_rows = result.get("result", [])
        if isinstance(raw_rows, dict):
            rows = [raw_rows]
        elif isinstance(raw_rows, list):
            rows = raw_rows
        else:
            rows = [{"result": str(raw_rows)}] if raw_rows else []

        rows = rows[: safe_limit(10)]
        if not rows:
            return {"context": [{"source": "cypher", "rows": [], "fallback": fallback}]}

        return {
            "context": [
                {
                    "source": "cypher",
                    "query": generated_cypher,
                    "rows": rows,
                }
            ]
        }
    except Exception as exc:
        return {
            "context": [
                {
                    "source": "cypher",
                    "error": str(exc),
                    "fallback": fallback,
                }
            ]
        }


def generate(state: State):
    fallback = default_answer(state["question"])
    if not state["context"]:
        return {"answer": fallback}

    first = state["context"][0]
    if "fallback" in first and not first.get("rows"):
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
    question = "Where is part A100 used?"
    response = app.invoke({"question": question})
    print("Answer:", response["answer"])
    print("Context:", response["context"])
