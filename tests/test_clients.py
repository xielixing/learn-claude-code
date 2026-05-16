import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mini_agent.clients import (
    AnthropicModelClient,
    OpenAIChatModelClient,
    load_codex_env,
    load_claude_code_env,
    merged_env,
)
from mini_agent.messages import Message


class ClientConfigTest(unittest.TestCase):
    def test_loads_env_from_claude_code_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "settings.json").write_text(
                json.dumps(
                    {
                        "env": {
                            "ANTHROPIC_MODEL": "claude-from-settings",
                            "ANTHROPIC_AUTH_TOKEN": "token-from-settings",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "settings.local.json").write_text(
                json.dumps({"env": {"ANTHROPIC_MODEL": "claude-from-local"}}),
                encoding="utf-8",
            )

            env = load_claude_code_env(config_dir)

        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-from-local")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "token-from-settings")

    def test_process_env_overrides_claude_code_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "settings.json").write_text(
                json.dumps({"env": {"ANTHROPIC_MODEL": "claude-from-settings"}}),
                encoding="utf-8",
            )

            env = merged_env(
                environ={"ANTHROPIC_MODEL": "claude-from-process"},
                claude_config_dir=config_dir,
            )

        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-from-process")

    def test_loads_env_from_codex_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "auth.json").write_text(
                json.dumps({"OPENAI_API_KEY": "key-from-codex"}),
                encoding="utf-8",
            )
            (config_dir / "config.toml").write_text(
                '\n'.join(
                    [
                        'model_provider = "OpenAI"',
                        'model = "gpt-from-codex"',
                        '',
                        '[model_providers.OpenAI]',
                        'base_url = "http://localhost:8080"',
                        'wire_api = "responses"',
                    ]
                ),
                encoding="utf-8",
            )

            env = load_codex_env(config_dir)

        self.assertEqual(env["OPENAI_API_KEY"], "key-from-codex")
        self.assertEqual(env["OPENAI_MODEL"], "gpt-from-codex")
        self.assertEqual(env["OPENAI_BASE_URL"], "http://localhost:8080")
        self.assertEqual(env["OPENAI_WIRE_API"], "responses")
        self.assertEqual(env["CODEX_CONFIG_LOADED"], "1")

    def test_codex_config_overrides_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "config.toml").write_text(
                'model_provider = "OpenAI"\nmodel = "gpt-from-codex"\n',
                encoding="utf-8",
            )

            env = merged_env(
                environ={"OPENAI_MODEL": "gpt-from-process"},
                use_claude_code_config=False,
                codex_config_dir=config_dir,
            )

        self.assertEqual(env["OPENAI_MODEL"], "gpt-from-codex")

    def test_uses_top_level_claude_code_model_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "settings.json").write_text(
                json.dumps({"model": "claude-top-level"}),
                encoding="utf-8",
            )

            env = load_claude_code_env(config_dir)

        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-top-level")


class AnthropicModelClientTest(unittest.TestCase):
    def test_sends_anthropic_blocks_to_messages_api(self) -> None:
        fake_messages = FakeAnthropicMessages()
        client = AnthropicModelClient(
            model="claude-test",
            client=SimpleNamespace(messages=fake_messages),
        )

        reply = client.create(
            system="You are concise.",
            messages=[Message(role="user", content="hello")],
            tools=[{"name": "read_file", "description": "Read.", "input_schema": {"type": "object"}}],
        )

        self.assertEqual(fake_messages.kwargs["model"], "claude-test")
        self.assertEqual(fake_messages.kwargs["system"], "You are concise.")
        self.assertEqual(fake_messages.kwargs["messages"][0]["content"], "hello")
        self.assertEqual(fake_messages.kwargs["tools"][0]["name"], "read_file")
        self.assertEqual(reply.content[0]["text"], "done")
        self.assertEqual(reply.usage["input_tokens"], 3)


class OpenAIChatModelClientTest(unittest.TestCase):
    def test_translates_tools_and_tool_results(self) -> None:
        fake_completions = FakeOpenAICompletions()
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions),
        )
        client = OpenAIChatModelClient(
            model="gpt-test",
            wire_api="chat",
            client=fake_client,
            use_codex_config=False,
        )

        reply = client.create(
            system="Use tools.",
            messages=[
                Message(role="user", content="read it"),
                Message(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file text",
                        }
                    ],
                ),
            ],
            tools=[{"name": "read_file", "description": "Read.", "input_schema": {"type": "object"}}],
        )

        messages = fake_completions.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(fake_completions.kwargs["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(reply.content[0]["type"], "tool_use")
        self.assertEqual(reply.content[0]["input"]["path"], "README.md")
        self.assertEqual(reply.usage["output_tokens"], 2)

    def test_uses_responses_api_when_configured(self) -> None:
        fake_responses = FakeOpenAIResponses()
        fake_client = SimpleNamespace(responses=fake_responses)
        client = OpenAIChatModelClient(
            model="gpt-test",
            wire_api="responses",
            client=fake_client,
            use_codex_config=False,
        )

        reply = client.create(
            system="Use tools.",
            messages=[
                Message(role="user", content="read it"),
                Message(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file text",
                        }
                    ],
                ),
            ],
            tools=[{"name": "read_file", "description": "Read.", "input_schema": {"type": "object"}}],
        )

        self.assertEqual(fake_responses.kwargs["instructions"], "Use tools.")
        self.assertEqual(fake_responses.kwargs["input"][1]["type"], "function_call")
        self.assertEqual(fake_responses.kwargs["input"][2]["type"], "function_call_output")
        self.assertEqual(fake_responses.kwargs["tools"][0]["name"], "read_file")
        self.assertEqual(fake_responses.kwargs["max_output_tokens"], 1024)
        self.assertEqual(reply.content[0]["text"], "done")
        self.assertEqual(reply.content[1]["type"], "tool_use")
        self.assertEqual(reply.content[1]["input"]["path"], "README.md")
        self.assertEqual(reply.usage["output_tokens"], 2)


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(model_dump=lambda exclude_none=True: {"type": "text", "text": "done"})],
            usage=SimpleNamespace(
                model_dump=lambda exclude_none=True: {"input_tokens": 3, "output_tokens": 1}
            ),
        )


class FakeOpenAICompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path": "README.md"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output=[
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
            ],
            usage=SimpleNamespace(
                model_dump=lambda exclude_none=True: {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
