from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ToolFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def run_tool_use(self, block: dict[str, Any]) -> str:
        tool = self._tools[block["name"]]
        return tool.run(block.get("input", {}))


def read_file_tool(root: Path | str = ".") -> Tool:
    base = Path(root).resolve()

    def run(args: dict[str, Any]) -> str:
        path = (base / args["path"]).resolve()
        path.relative_to(base)
        return path.read_text(encoding="utf-8")

    return Tool(
        name="read_file",
        description="Read a text file under the project root.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        run=run,
    )
