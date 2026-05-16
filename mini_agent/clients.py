import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import ModelReply
from .messages import Message


def claude_code_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = environ or os.environ
    configured = env.get("CLAUDE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".claude"


def load_claude_code_env(config_dir: Path | str | None = None) -> dict[str, str]:
    base = Path(config_dir).expanduser() if config_dir else claude_code_config_dir()
    loaded: dict[str, str] = {}

    for name in ("settings.json", "settings.local.json"):
        path = base / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        settings_env = data.get("env")
        if not isinstance(settings_env, dict):
            settings_env = {}
        if data.get("model") and "ANTHROPIC_MODEL" not in settings_env:
            loaded["ANTHROPIC_MODEL"] = str(data["model"])
        for key, value in settings_env.items():
            if value is not None:
                loaded[str(key)] = str(value)

    return loaded


def codex_config_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = environ or os.environ
    configured = env.get("CODEX_HOME") or env.get("CODEX_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def load_codex_env(config_dir: Path | str | None = None) -> dict[str, str]:
    base = Path(config_dir).expanduser() if config_dir else codex_config_dir()
    loaded: dict[str, str] = {}

    auth_path = base / "auth.json"
    if auth_path.exists():
        data = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        api_key = data.get("OPENAI_API_KEY")
        if api_key:
            loaded["OPENAI_API_KEY"] = str(api_key)

    config_path = base / "config.toml"
    if config_path.exists():
        data = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
        if data.get("model"):
            loaded["OPENAI_MODEL"] = str(data["model"])

        provider_name = data.get("model_provider")
        providers = data.get("model_providers", {})
        provider = providers.get(provider_name, {}) if isinstance(providers, dict) else {}
        if isinstance(provider, dict):
            if provider.get("base_url"):
                loaded["OPENAI_BASE_URL"] = str(provider["base_url"])
            if provider.get("wire_api"):
                loaded["OPENAI_WIRE_API"] = str(provider["wire_api"])

    if loaded:
        loaded["CODEX_CONFIG_LOADED"] = "1"
    return loaded


def merged_env(
    *,
    environ: Mapping[str, str] | None = None,
    use_codex_config: bool = True,
    codex_config_dir: Path | str | None = None,
    use_claude_code_config: bool = True,
    claude_config_dir: Path | str | None = None,
) -> dict[str, str]:
    process_env = dict(environ or os.environ)
    claude_env = load_claude_code_env(claude_config_dir) if use_claude_code_config else {}
    codex_env = load_codex_env(codex_config_dir) if use_codex_config else {}

    return {**claude_env, **process_env, **codex_env}


@dataclass
class AnthropicModelClient:
    model: str | None = None
    max_tokens: int = 1024
    client: Any | None = None
    environ: Mapping[str, str] | None = None
    use_codex_config: bool = True
    codex_config_dir: Path | str | None = None
    use_claude_code_config: bool = True
    claude_config_dir: Path | str | None = None

    def __post_init__(self) -> None:
        env = merged_env(
            environ=self.environ,
            use_codex_config=self.use_codex_config,
            codex_config_dir=self.codex_config_dir,
            use_claude_code_config=self.use_claude_code_config,
            claude_config_dir=self.claude_config_dir,
        )
        self.model = self.model or env.get("ANTHROPIC_MODEL")

        if self.client is None:
            self.client = _new_anthropic_client(env)

    def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        if not self.model:
            raise ValueError("Anthropic model is not configured. Set --model or ANTHROPIC_MODEL.")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [_anthropic_message(message) for message in messages],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        return ModelReply(
            content=[_dump_sdk_object(block) for block in response.content],
            usage=_usage_dict(getattr(response, "usage", None)),
        )


@dataclass
class OpenAIChatModelClient:
    model: str | None = None
    max_tokens: int | None = 1024
    wire_api: str | None = None
    client: Any | None = None
    environ: Mapping[str, str] | None = None
    use_codex_config: bool = True
    codex_config_dir: Path | str | None = None
    use_claude_code_config: bool = True
    claude_config_dir: Path | str | None = None

    def __post_init__(self) -> None:
        env = merged_env(
            environ=self.environ,
            use_codex_config=self.use_codex_config,
            codex_config_dir=self.codex_config_dir,
            use_claude_code_config=self.use_claude_code_config,
            claude_config_dir=self.claude_config_dir,
        )
        self.model = self.model or env.get("OPENAI_MODEL")
        self.wire_api = (self.wire_api or env.get("OPENAI_WIRE_API") or "chat").lower()

        if self.client is None:
            self.client = _new_openai_client(env)

    def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        if not self.model:
            raise ValueError("OpenAI model is not configured. Set --model or OPENAI_MODEL.")

        if self.wire_api == "responses":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": _openai_response_input(messages),
            }
            if system:
                kwargs["instructions"] = system
            if tools:
                kwargs["tools"] = [_openai_response_tool(tool) for tool in tools]
            if self.max_tokens is not None:
                kwargs["max_output_tokens"] = self.max_tokens

            response = self.client.responses.create(**kwargs)
            return ModelReply(
                content=_openai_response_content(response),
                usage=_usage_dict(getattr(response, "usage", None)),
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = [_openai_tool(tool) for tool in tools]
        if self.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return ModelReply(
            content=_openai_reply_content(message),
            usage=_openai_usage_dict(getattr(response, "usage", None)),
        )


def _new_anthropic_client(env: Mapping[str, str]) -> Any:
    api_key = env.get("ANTHROPIC_API_KEY")
    auth_token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not api_key and not auth_token:
        raise ValueError(
            "Anthropic credentials are not configured. Set ANTHROPIC_API_KEY, "
            "ANTHROPIC_AUTH_TOKEN, or put them in Claude Code settings env."
        )

    from anthropic import Anthropic

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if auth_token:
        kwargs["auth_token"] = auth_token
    if env.get("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = env["ANTHROPIC_BASE_URL"]
    if env.get("API_TIMEOUT_MS"):
        kwargs["timeout"] = int(env["API_TIMEOUT_MS"]) / 1000
    headers = _json_object_env(env.get("ANTHROPIC_CUSTOM_HEADERS"))
    if headers:
        kwargs["default_headers"] = {str(key): str(value) for key, value in headers.items()}

    return Anthropic(**kwargs)


def _new_openai_client(env: Mapping[str, str]) -> Any:
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI credentials are not configured. Set OPENAI_API_KEY or put it in Claude Code settings env."
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if env.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = env["OPENAI_BASE_URL"]
    if env.get("OPENAI_ORG_ID"):
        kwargs["organization"] = env["OPENAI_ORG_ID"]
    if env.get("OPENAI_PROJECT"):
        kwargs["project"] = env["OPENAI_PROJECT"]

    return OpenAI(**kwargs)


def _json_object_env(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _anthropic_message(message: Message) -> dict[str, Any]:
    if message.role == "system":
        raise ValueError("System messages must be passed via the system argument.")
    return {"role": message.role, "content": message.content}


def _dump_sdk_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    dumped = _dump_sdk_object(usage)
    return {key: value for key, value in dumped.items() if isinstance(value, int)}


def _openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if system:
        converted.append({"role": "system", "content": system})

    for message in messages:
        if message.role == "system":
            converted.append({"role": "system", "content": _content_text(message.content)})
        elif message.role == "user":
            converted.extend(_openai_user_messages(message.content))
        elif message.role == "assistant":
            converted.append(_openai_assistant_message(message.content))
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    return converted


def _openai_user_messages(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"role": "user", "content": _content_text(content)}]

    converted: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(_content_text(block))
            continue

        block_type = block.get("type")
        if block_type == "tool_result":
            if text_parts:
                converted.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts = []
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": _content_text(block.get("content", "")),
                }
            )
        elif block_type == "text":
            text_parts.append(str(block.get("text", "")))
        else:
            text_parts.append(json.dumps(block, ensure_ascii=False))

    if text_parts:
        converted.append({"role": "user", "content": "\n".join(text_parts)})
    return converted


def _openai_assistant_message(content: Any) -> dict[str, Any]:
    if not isinstance(content, list):
        return {"role": "assistant", "content": _content_text(content)}

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(_content_text(block))
            continue

        block_type = block.get("type")
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif block_type == "text":
            text_parts.append(str(block.get("text", "")))

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        },
    }


def _openai_response_input(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            converted.extend(_openai_response_user_items(message.content))
        elif message.role == "assistant":
            converted.extend(_openai_response_assistant_items(message.content))
        elif message.role == "system":
            converted.append({"role": "system", "content": _content_text(message.content)})
        else:
            raise ValueError(f"Unsupported message role: {message.role}")
    return converted


def _openai_response_user_items(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"role": "user", "content": _content_text(content)}]

    converted: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(_content_text(block))
            continue

        if block.get("type") == "tool_result":
            if text_parts:
                converted.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts = []
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": block["tool_use_id"],
                    "output": _content_text(block.get("content", "")),
                }
            )
        elif block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        else:
            text_parts.append(json.dumps(block, ensure_ascii=False))

    if text_parts:
        converted.append({"role": "user", "content": "\n".join(text_parts)})
    return converted


def _openai_response_assistant_items(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"role": "assistant", "content": _content_text(content)}]

    converted: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(_content_text(block))
            continue

        if block.get("type") == "tool_use":
            if text_parts:
                converted.append({"role": "assistant", "content": "\n".join(text_parts)})
                text_parts = []
            converted.append(
                {
                    "type": "function_call",
                    "call_id": block["id"],
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                }
            )
        elif block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))

    if text_parts:
        converted.append({"role": "assistant", "content": "\n".join(text_parts)})
    return converted


def _openai_response_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object"}),
    }


def _openai_response_content(response: Any) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        dumped = _dump_sdk_object(item)
        item_type = dumped.get("type")
        if item_type == "message":
            for block in dumped.get("content", []):
                block_type = block.get("type") if isinstance(block, dict) else None
                if block_type in {"output_text", "text"} and block.get("text"):
                    content.append({"type": "text", "text": block["text"]})
        elif item_type == "function_call":
            arguments = dumped.get("arguments") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"_raw": arguments}
            content.append(
                {
                    "type": "tool_use",
                    "id": dumped.get("call_id") or dumped.get("id"),
                    "name": dumped["name"],
                    "input": parsed_arguments,
                }
            )

    if not any(block.get("type") == "text" for block in content):
        output_text = getattr(response, "output_text", None)
        if output_text:
            content.insert(0, {"type": "text", "text": output_text})
    return content


def _openai_reply_content(message: Any) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    text = getattr(message, "content", None)
    if text:
        content.append({"type": "text", "text": text})

    for call in getattr(message, "tool_calls", None) or []:
        arguments = getattr(call.function, "arguments", "") or "{}"
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"_raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": parsed_arguments,
            }
        )

    return content


def _openai_usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
