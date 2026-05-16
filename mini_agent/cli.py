import argparse
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from mini_agent import ToolRegistry
from mini_agent.tools import read_file_tool
from .clients import AnthropicModelClient, OpenAIChatModelClient, merged_env
from .engine import ModelReply, QueryEngine


class EchoModel:
    def create(
        self,
        *,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        return ModelReply(content=[{"type": "text", "text": messages[-1].content}])


def resolve_provider(args: argparse.Namespace, environ: Mapping[str, str] | None = None) -> str:
    if args.provider != "auto":
        return args.provider

    env = merged_env(
        environ=environ,
        use_codex_config=not getattr(args, "no_codex_config", False),
        codex_config_dir=getattr(args, "codex_config_dir", None),
        use_claude_code_config=not getattr(args, "no_claude_code_config", False),
        claude_config_dir=getattr(args, "claude_config_dir", None),
    )
    if env.get("CODEX_CONFIG_LOADED") and env.get("OPENAI_API_KEY"):
        return "openai"

    has_anthropic_credentials = any(
        env.get(name)
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    )
    if has_anthropic_credentials:
        return "anthropic"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    return "echo"


def build_model(args: argparse.Namespace, provider: str) -> Any:
    if provider == "echo":
        return EchoModel()
    if provider == "anthropic":
        return AnthropicModelClient(
            model=args.model,
            max_tokens=args.max_tokens,
            use_codex_config=not getattr(args, "no_codex_config", False),
            codex_config_dir=getattr(args, "codex_config_dir", None),
            use_claude_code_config=not getattr(args, "no_claude_code_config", False),
            claude_config_dir=getattr(args, "claude_config_dir", None),
        )
    if provider == "openai":
        return OpenAIChatModelClient(
            model=args.model,
            max_tokens=args.max_tokens,
            use_codex_config=not getattr(args, "no_codex_config", False),
            codex_config_dir=getattr(args, "codex_config_dir", None),
            use_claude_code_config=not getattr(args, "no_claude_code_config", False),
            claude_config_dir=getattr(args, "claude_config_dir", None),
        )
    raise ValueError(f"Unsupported provider: {provider}")

def build_tools() -> ToolRegistry:
    return ToolRegistry([read_file_tool(".")])

def format_usage(usage: dict[str, int]) -> str:
    keys = ["input_tokens", "output_tokens", "total_tokens"]
    parts = [f"{key}={usage[key]}" for key in keys if key in usage]
    return " ".join(parts)

def print_events(events: list[dict[str, Any]], output: TextIO = sys.stdout) -> None:
    for event in events:
        if event["type"] == "assistant":
            for block in event["content"]:
                if block.get("type") == "text":
                    print(block["text"], file=output)
                elif block.get("type") == "tool_use":
                    print(f"[tool] {block['name']} {block.get('input', {})}", file=output)
            usage_text = format_usage(event.get("usage", {}))
            if usage_text:
                print(f"[usage] {usage_text}", file=output)
        elif event["type"] == "tool_results":
            for block in event["content"]:
                print(f"[tool result] {block['content']}", file=output)


def run_once(engine: QueryEngine, prompt: str, output: TextIO = sys.stdout) -> None:
    events = engine.submit_message(prompt)
    print_events(events, output)


def run_repl(
    engine: QueryEngine,
    *,
    provider: str,
    model: str | None,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> None:
    print("mini-agent interactive", file=output)
    print("Type /help for commands, /exit to quit.", file=output)

    while True:
        try:
            prompt = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=output)
            return

        if not prompt:
            continue
        if prompt.startswith("/"):
            if handle_command(prompt, engine, provider=provider, model=model, output=output):
                return
            continue

        try:
            run_once(engine, prompt, output)
        except Exception as exc:
            print(f"mini-agent: {exc}", file=output)


def handle_command(
    command: str,
    engine: QueryEngine,
    *,
    provider: str,
    model: str | None,
    output: TextIO = sys.stdout,
) -> bool:
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        print("/help     Show commands.", file=output)
        print("/model    Show provider and model.", file=output)
        print("/history  Show message count.", file=output)
        print("/clear    Clear conversation context.", file=output)
        print("/exit     Quit.", file=output)
        return False
    if command == "/model":
        configured_model = model or getattr(engine.model, "model", None) or "(default)"
        print(f"provider={provider} model={configured_model}", file=output)
        return False
    if command == "/history":
        print(f"{len(engine.messages)} messages in context.", file=output)
        return False
    if command == "/clear":
        engine.messages.clear()
        print("context cleared.", file=output)
        return False

    print(f"unknown command: {command}", file=output)
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["auto", "echo", "anthropic", "openai"], default="auto")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--system", default="")
    parser.add_argument("--codex-config-dir")
    parser.add_argument(
        "--no-codex-config",
        action="store_true",
        help="Do not load env values from local Codex config.",
    )
    parser.add_argument("--claude-config-dir")
    parser.add_argument(
        "--no-claude-code-config",
        action="store_true",
        help="Do not load env values from Claude Code settings.",
    )
    parser.add_argument("prompt", nargs="?")
    args = parser.parse_args(argv)

    try:
        provider = resolve_provider(args)
        engine = QueryEngine(
            model=build_model(args, provider),
            tools=build_tools(),
            system_prompt=args.system
        )
        if args.prompt is None:
            run_repl(engine, provider=provider, model=args.model)
        else:
            run_once(engine, args.prompt)
    except Exception as exc:
        print(f"mini-agent: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
