from langchain_core.messages import AIMessage, HumanMessage

from rag.memory import window


def test_window_keeps_last_n():
    msgs = [HumanMessage(str(i)) if i % 2 == 0 else AIMessage(str(i)) for i in range(10)]
    out = window(msgs, 6)
    assert [m.content for m in out] == ["4", "5", "6", "7", "8", "9"]


def test_window_shorter_than_n():
    msgs = [HumanMessage("a")]
    assert window(msgs, 6) == msgs


def test_window_zero():
    assert window([HumanMessage("a")], 0) == []
