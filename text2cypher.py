import os
from operator import add
from typing import Annotated, List, Optional, Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

from langgraph.graph import StateGraph, END

# -------------------------
# ENV / MODEL / GRAPH SETUP
# -------------------------
# Requires:
# OPENAI_API_KEY
# NEO4J_URI
# NEO4J_USERNAME
# NEO4J_PASSWORD

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

graph = Neo4jGraph()
graph.refresh_schema()

NO_RESULTS = "I couldn't find any relevant information in the database."
NON_DOMAIN_MSG = "This question is not about movies/BOM domain data available in this graph."

MAX_RETRIES = 3

# -------------------------
# STATE
# -------------------------
class InputState(TypedDict):
    question: str


class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict] | str
    steps: Annotated[List[str], add]
    retry_count: int


class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str


# -------------------------
# GUARDRAILS
# -------------------------
guardrails_system = """
You are a classifier.
If the question is related to movies / film data / cast / directors / genres, output "movie".
Otherwise output "end".
Return only one of: movie, end.
"""

guardrails_prompt = ChatPromptTemplate.from_messages(
    [("system", guardrails_system), ("human", "{question}")]
)

class GuardrailsOutput(BaseModel):
    decision: Literal["movie", "end"]

guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

def guardrails(state: InputState) -> OverallState:
    result = guardrails_chain.invoke({"question": state["question"]})
    if result.decision == "end":
        return {
            "question": state["question"],
            "next_action": "end",
            "database_records": NON_DOMAIN_MSG,
            "steps": ["guardrails"],
            "retry_count": 0,
            "cypher_statement": "",
            "cypher_errors": [],
        }
    return {
        "question": state["question"],
        "next_action": "movie",
        "database_records": [],
        "steps": ["guardrails"],
        "retry_count": 0,
        "cypher_statement": "",
        "cypher_errors": [],
    }


# -------------------------
# FEW-SHOT EXAMPLES
# -------------------------
examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a) AS count",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a:Person) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m) AS count",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: \"Schindler's List\"})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Neo4jVector,
    k=4,
    input_keys=["question"],
)

# -------------------------
# TEXT -> CYPHER
# -------------------------
text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Convert user question to Cypher using ONLY provided schema. "
         "No markdown. Output Cypher only."),
        ("human",
         """Schema:
{schema}

Few-shot:
{fewshot_examples}

Question: {question}
Cypher:"""),
    ]
)
text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

def generate_cypher(state: OverallState) -> OverallState:
    nl = "\n"
    selected = example_selector.select_examples({"question": state["question"]})
    fewshot = (nl * 2).join([f"Question: {e['question']}{nl}Cypher: {e['query']}" for e in selected])

    cypher = text2cypher_chain.invoke(
        {
            "question": state["question"],
            "fewshot_examples": fewshot,
            "schema": graph.schema,
        }
    ).strip()

    return {
        **state,
        "cypher_statement": cypher,
        "next_action": "validate_cypher",
        "steps": ["generate_cypher"],
    }


# -------------------------
# VALIDATION
# -------------------------
validate_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Cypher expert validator."),
        ("human",
         """Check the Cypher against schema and question.
Return concise errors list when present.

Schema:
{schema}

Question:
{question}

Cypher:
{cypher}"""),
    ]
)

class ValidateCypherOutput(BaseModel):
    errors: Optional[List[str]] = Field(default=None)

validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)

corrector_schema = [
    Schema(rel["start"], rel["type"], rel["end"])
    for rel in graph.structured_schema.get("relationships", [])
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

def validate_cypher(state: OverallState) -> OverallState:
    errors = []
    cypher = state["cypher_statement"]

    # 1) syntax check
    try:
        graph.query(f"EXPLAIN {cypher}")
    except CypherSyntaxError as e:
        errors.append(str(e))

    # 2) relationship direction correction
    corrected = cypher_query_corrector(cypher)
    if not corrected:
        errors.append("Generated Cypher does not fit graph schema.")
        corrected = cypher

    # 3) LLM semantic/schema check
    llm_check = validate_cypher_chain.invoke(
        {"schema": graph.schema, "question": state["question"], "cypher": corrected}
    )
    if llm_check.errors:
        errors.extend(llm_check.errors)

    if errors:
        if state["retry_count"] >= MAX_RETRIES:
            return {
                **state,
                "cypher_statement": corrected,
                "cypher_errors": errors,
                "next_action": "end",
                "database_records": f"Failed after {MAX_RETRIES} correction attempts. Last errors: {errors}",
                "steps": ["validate_cypher"],
            }
        return {
            **state,
            "cypher_statement": corrected,
            "cypher_errors": errors,
            "next_action": "correct_cypher",
            "steps": ["validate_cypher"],
        }

    return {
        **state,
        "cypher_statement": corrected,
        "cypher_errors": [],
        "next_action": "execute_cypher",
        "steps": ["validate_cypher"],
    }


# -------------------------
# CORRECTION
# -------------------------
correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You fix Cypher. Output only corrected Cypher."),
        ("human",
         """Schema:
{schema}

Question:
{question}

Cypher:
{cypher}

Errors:
{errors}

Return corrected Cypher only:"""),
    ]
)
correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()

def correct_cypher(state: OverallState) -> OverallState:
    corrected = correct_cypher_chain.invoke(
        {
            "schema": graph.schema,
            "question": state["question"],
            "cypher": state["cypher_statement"],
            "errors": state["cypher_errors"],
        }
    ).strip()

    return {
        **state,
        "cypher_statement": corrected,
        "retry_count": state["retry_count"] + 1,
        "next_action": "validate_cypher",
        "steps": ["correct_cypher"],
    }


# -------------------------
# EXECUTION
# -------------------------
def execute_cypher(state: OverallState) -> OverallState:
    try:
        records = graph.query(state["cypher_statement"])
        data = records if records else NO_RESULTS
    except Exception as e:
        data = f"Execution error: {e}"

    return {
        **state,
        "database_records": data,
        "next_action": "end",
        "steps": ["execute_cypher"],
    }


# -------------------------
# FINAL ANSWER
# -------------------------
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant."),
        ("human", "Question: {question}\nResults: {results}\nAnswer directly."),
    ]
)
final_chain = final_prompt | llm | StrOutputParser()

def generate_final_answer(state: OverallState) -> OutputState:
    if isinstance(state["database_records"], str) and (
        state["database_records"] == NON_DOMAIN_MSG
        or state["database_records"].startswith("Failed after")
        or state["database_records"].startswith("Execution error")
    ):
        answer = state["database_records"]
    else:
        answer = final_chain.invoke(
            {"question": state["question"], "results": state["database_records"]}
        )

    return {
        "answer": answer,
        "steps": state.get("steps", []),
        "cypher_statement": state.get("cypher_statement", ""),
    }


# -------------------------
# ROUTING
# -------------------------
def route_after_guardrails(state: OverallState) -> Literal["generate_cypher", "generate_final_answer"]:
    return "generate_cypher" if state["next_action"] == "movie" else "generate_final_answer"

def route_after_validate(state: OverallState) -> Literal["correct_cypher", "execute_cypher", "generate_final_answer"]:
    if state["next_action"] == "correct_cypher":
        return "correct_cypher"
    if state["next_action"] == "execute_cypher":
        return "execute_cypher"
    return "generate_final_answer"


# -------------------------
# LANGGRAPH WORKFLOW
# -------------------------
builder = StateGraph(OverallState, input=InputState, output=OutputState)

builder.add_node("guardrails", guardrails)
builder.add_node("generate_cypher", generate_cypher)
builder.add_node("validate_cypher", validate_cypher)
builder.add_node("correct_cypher", correct_cypher)
builder.add_node("execute_cypher", execute_cypher)
builder.add_node("generate_final_answer", generate_final_answer)

builder.set_entry_point("guardrails")

builder.add_conditional_edges(
    "guardrails",
    route_after_guardrails,
    {
        "generate_cypher": "generate_cypher",
        "generate_final_answer": "generate_final_answer",
    },
)

builder.add_edge("generate_cypher", "validate_cypher")

builder.add_conditional_edges(
    "validate_cypher",
    route_after_validate,
    {
        "correct_cypher": "correct_cypher",
        "execute_cypher": "execute_cypher",
        "generate_final_answer": "generate_final_answer",
    },
)

builder.add_edge("correct_cypher", "validate_cypher")
builder.add_edge("execute_cypher", "generate_final_answer")
builder.add_edge("generate_final_answer", END)

app = builder.compile()


# -------------------------
# USAGE EXAMPLE
# -------------------------
if __name__ == "__main__":
    question = "Which actors played in the movie Casino?"
    result = app.invoke({"question": question})
    print("Answer:", result["answer"])
    print("Cypher:", result["cypher_statement"])
    print("Steps:", result["steps"])
