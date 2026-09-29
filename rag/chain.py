"""RAG pipeline as a LangGraph state graph.

    START → condense → retrieve ─┬─ (docs found) → generate  → END
                                 └─ (no docs)    → not_found → END

Conversation memory lives in the graph state (`messages`) and is persisted per
thread by the checkpointer; nodes see only the last `memory_window` messages.
"""

from collections.abc import Iterator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag.config import DEFAULT_MEMORY_WINDOW
from rag.memory import window

NOT_FOUND = "I couldn't find this in the selected documents."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the question so it "
            "can be understood without the chat history. Do NOT answer it. "
            "If it is already standalone, return it unchanged. Output only the question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document Q&A assistant. Answer the user's question using ONLY the context "
            "below, which consists of excerpts from the user's documents.\n\n"
            "Rules:\n"
            "- Do not use outside knowledge or make assumptions beyond the context.\n"
            f"- If the context does not contain the answer, reply exactly: \"{NOT_FOUND}\"\n"
            "- Cite every factual claim inline using the source label, e.g. [report.pdf, p. 4] "
            "or [notes.txt] when there is no page.\n"
            "- Be concise and well structured.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


class RAGState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation, checkpointed
    memory_window: int
    question: str
    standalone_question: str
    docs: list[Document]
    answer: str


def source_label(doc: Document) -> str:
    page = doc.metadata.get("page", -1)
    name = doc.metadata.get("file_name", "unknown")
    return f"{name}, p. {page}" if page and page > 0 else name


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source {i} | {source_label(d)}]\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


def _history(state: RAGState) -> list[BaseMessage]:
    """Prior turns (excluding the current question), trimmed to the memory window."""
    return window(state["messages"][:-1], state.get("memory_window", DEFAULT_MEMORY_WINDOW))


def build_graph(llm: BaseChatModel, retriever: BaseRetriever, checkpointer: BaseCheckpointSaver | None = None):
    def condense(state: RAGState) -> RAGState:
        history = _history(state)
        if not history:
            return {"standalone_question": state["question"]}
        standalone = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
            {"history": history, "question": state["question"]}
        )
        return {"standalone_question": standalone.strip() or state["question"]}

    def retrieve(state: RAGState) -> RAGState:
        return {"docs": retriever.invoke(state["standalone_question"])}

    def generate(state: RAGState) -> RAGState:
        answer = (ANSWER_PROMPT | llm | StrOutputParser()).invoke(
            {
                "context": format_context(state["docs"]),
                "history": _history(state),
                "question": state["standalone_question"],
            }
        )
        return {"answer": answer, "messages": [AIMessage(answer)]}

    def not_found(state: RAGState) -> RAGState:
        return {"answer": NOT_FOUND, "messages": [AIMessage(NOT_FOUND)]}

    graph = StateGraph(RAGState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("not_found", not_found)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_conditional_edges(
        "retrieve", lambda s: "generate" if s.get("docs") else "not_found", ["generate", "not_found"]
    )
    graph.add_edge("generate", END)
    graph.add_edge("not_found", END)
    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def stream_answer(
    graph, question: str, thread_id: str, memory_window: int, result: dict
) -> Iterator[str]:
    """Stream answer tokens from the `generate` node; node outputs (e.g. docs) are collected into `result`."""
    inputs = {"question": question, "messages": [HumanMessage(question)], "memory_window": memory_window}
    streamed = False
    for mode, payload in graph.stream(
        inputs, thread_config(thread_id), stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node, update in payload.items():
                update = update or {}
                result.update(update)
                # not_found never streams; generate only needs this if the model didn't stream tokens.
                if node == "not_found" or (node == "generate" and not streamed):
                    yield update["answer"]
        else:
            chunk, meta = payload
            # Only token chunks — the final AIMessage written to state is also emitted here.
            if meta.get("langgraph_node") == "generate" and isinstance(chunk, AIMessageChunk) and chunk.text:
                streamed = True
                yield chunk.text
