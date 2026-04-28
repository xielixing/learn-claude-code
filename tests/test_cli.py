import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from mini_agent.cli import EchoModel, handle_command, resolve_provider, run_once, run_repl
from mini_agent.engine import QueryEngine


class CliTest(unittest.TestCase):
    def test_run_once_prints_reply(self) -> None:
        engine = QueryEngine(model=EchoModel())
        output = io.StringIO()

        run_once(engine, "hello", output)

        self.assertEqual(output.getvalue().strip(), "hello")

    def test_repl_keeps_context_until_clear(self) -> None:
        engine = QueryEngine(model=EchoModel())
        output = io.StringIO()
        inputs = iter(["hello", "/history", "/clear", "/history", "/exit"])

        run_repl(
            engine,
            provider="echo",
            model=None,
            input_fn=lambda prompt: next(inputs),
            output=output,
        )

        text = output.getvalue()
        self.assertIn("2 messages in context.", text)
        self.assertIn("context cleared.", text)
        self.assertIn("0 messages in context.", text)

    def test_handle_model_command_uses_client_model(self) -> None:
        engine = QueryEngine(model=EchoModel())
        engine.model.model = "echo-test"
        output = io.StringIO()

        should_exit = handle_command(
            "/model",
            engine,
            provider="echo",
            model=None,
            output=output,
        )

        self.assertFalse(should_exit)
        self.assertEqual(output.getvalue().strip(), "provider=echo model=echo-test")

    def test_auto_provider_prefers_claude_code_anthropic_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "settings.json").write_text(
                json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "token"}}),
                encoding="utf-8",
            )
            args = Namespace(
                provider="auto",
                no_claude_code_config=False,
                claude_config_dir=str(config_dir),
            )

            provider = resolve_provider(args, environ={})

        self.assertEqual(provider, "anthropic")

    def test_auto_provider_falls_back_to_echo_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                provider="auto",
                no_claude_code_config=False,
                claude_config_dir=tmp,
            )

            provider = resolve_provider(args, environ={})

        self.assertEqual(provider, "echo")


if __name__ == "__main__":
    unittest.main()
