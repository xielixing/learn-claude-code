import unittest
from typing import Any

from mini_agent import ModelReply, QueryEngine, Tool, ToolRegistry


class ScriptedModel:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        return self.replies.pop(0)


class AgentLoopTest(unittest.TestCase):
    def test_stops_when_reply_has_no_tool_use(self) -> None:
        model = ScriptedModel([ModelReply([{"type": "text", "text": "done"}])])
        engine = QueryEngine(model=model)

        events = engine.submit_message("hello")

        self.assertEqual(events[-1]["content"][0]["text"], "done")
        self.assertEqual([message.role for message in engine.messages], ["user", "assistant"])

    def test_runs_tool_and_feeds_result_back(self) -> None:
        tool = Tool(
            name="read_note",
            description="Read a note.",
            input_schema={"type": "object"},
            run=lambda args: "note content",
        )
        model = ScriptedModel(
            [
                ModelReply([{"type": "tool_use", "id": "1", "name": "read_note", "input": {}}]),
                ModelReply([{"type": "text", "text": "note content"}]),
            ]
        )
        engine = QueryEngine(model=model, tools=ToolRegistry([tool]))

        events = engine.submit_message("read the note")

        self.assertEqual(events[1]["content"][0]["content"], "note content")
        self.assertEqual(model.calls[1]["messages"][-1].content[0]["type"], "tool_result")

    def test_raises_when_max_turns_is_reached(self) -> None:
        model = ScriptedModel(
            [ModelReply([{"type": "tool_use", "id": "1", "name": "again", "input": {}}])]
        )
        tool = Tool("again", "Loop forever.", {"type": "object"}, lambda args: "again")
        engine = QueryEngine(model=model, tools=ToolRegistry([tool]), max_turns=1)

        with self.assertRaisesRegex(RuntimeError, "max_turns"):
            engine.submit_message("keep going")


if __name__ == "__main__":
    unittest.main()
