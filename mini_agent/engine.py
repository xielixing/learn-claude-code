from dataclasses import dataclass, field
from typing import Any, Protocol

from .messages import Message
from .tools import ToolRegistry


@dataclass(frozen=True)
class ModelReply:
    content: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


class ModelClient(Protocol):
    def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelReply: ...


@dataclass
class QueryEngine:
    model: ModelClient
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system_prompt: str = ""
    max_turns: int = 5
    messages: list[Message] = field(default_factory=list)

    def submit_message(self, prompt: str) -> list[dict[str, Any]]:
        self.messages.append(Message(role="user", content=prompt))
        return self.query_loop()

    def query_loop(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        for _ in range(self.max_turns):
            reply = self.model.create(
                system=self.system_prompt,
                messages=list(self.messages),
                tools=self.tools.schemas(),
            )
            self.messages.append(Message(role="assistant", content=reply.content))
            events.append({"type": "assistant", "content": reply.content, "usage": reply.usage})

            tool_uses = [block for block in reply.content if block.get("type") == "tool_use"]
            if not tool_uses:
                return events

            results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": self.tools.run_tool_use(block),
                }
                for block in tool_uses
            ]
            self.messages.append(Message(role="user", content=results))
            events.append({"type": "tool_results", "content": results})

        raise RuntimeError("max_turns reached")
