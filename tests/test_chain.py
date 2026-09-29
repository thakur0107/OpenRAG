from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver

from rag.chain import NOT_FOUND, build_graph, stream_answer, thread_config


class StubRetriever(BaseRetriever):
    docs: list[Document]
    queries: list[str] = []

    def _get_relevant_documents(self, query, *, run_manager=None):
        self.queries.append(query)
        return self.docs


DOC = Document(page_content="Cats sleep 16 hours.", metadata={"file_name": "cats.txt", "page": -1})


def _ask(graph, question, thread_id="t1", memory_window=6):
    result = {}
    text = "".join(stream_answer(graph, question, thread_id, memory_window, result))
    return text, result


def test_first_question_skips_condense_and_is_checkpointed():
    saver = InMemorySaver()
    retriever = StubRetriever(docs=[DOC], queries=[])
    graph = build_graph(FakeListChatModel(responses=["16 hours [cats.txt]."]), retriever, saver)
    text, result = _ask(graph, "How long do cats sleep?")
    assert retriever.queries == ["How long do cats sleep?"]
    assert text == "16 hours [cats.txt]." and result["docs"] == [DOC]
    msgs = graph.get_state(thread_config("t1")).values["messages"]
    assert [(m.type, m.content) for m in msgs] == [
        ("human", "How long do cats sleep?"),
        ("ai", "16 hours [cats.txt]."),
    ]


def test_follow_up_uses_checkpointed_history():
    saver = InMemorySaver()
    retriever = StubRetriever(docs=[DOC], queries=[])
    llm = FakeListChatModel(responses=["Cats are mammals.", "How long do cats sleep?", "16 hours."])
    # Rebuild the graph per turn (as the app does) — memory survives via the shared checkpointer.
    _ask(build_graph(llm, retriever, saver), "Tell me about cats")
    text, _ = _ask(build_graph(llm, retriever, saver), "how long do they sleep?")
    assert retriever.queries[-1] == "How long do cats sleep?"  # condensed from history
    assert text == "16 hours."
    assert len(saver.get_tuple(thread_config("t1")).checkpoint["channel_values"]["messages"]) == 4


def test_new_thread_has_no_history():
    saver = InMemorySaver()
    retriever = StubRetriever(docs=[DOC], queries=[])
    llm = FakeListChatModel(responses=["a1", "a2"])  # no condense call expected on either thread
    _ask(build_graph(llm, retriever, saver), "q1", thread_id="t1")
    _ask(build_graph(llm, retriever, saver), "q2", thread_id="t2")
    assert retriever.queries == ["q1", "q2"]


def test_memory_window_limits_history():
    saver = InMemorySaver()
    retriever = StubRetriever(docs=[DOC], queries=[])
    graph = build_graph(FakeListChatModel(responses=["a1", "a2"]), retriever, saver)
    _ask(graph, "q1")
    _ask(graph, "q2", memory_window=0)  # window 0 → no history → no condense call
    assert retriever.queries == ["q1", "q2"]


def test_no_docs_routes_to_not_found_without_calling_llm():
    graph = build_graph(FakeListChatModel(responses=[]), StubRetriever(docs=[], queries=[]), InMemorySaver())
    text, result = _ask(graph, "q?")
    assert text == NOT_FOUND and result["docs"] == []
    assert graph.get_state(thread_config("t1")).values["messages"][-1].content == NOT_FOUND
