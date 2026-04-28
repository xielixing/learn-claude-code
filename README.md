# learn-claude-code

一个极小的 Python 学习仓库，用来逐步实现一个类似 Claude Code agent loop 的最小 agent 框架。

当前保留几个核心概念：

- `QueryEngine`: 一次会话的入口，保存消息历史。
- `query_loop`: 模型回复、工具执行、工具结果回填的循环。
- `ToolRegistry`: 注册和执行工具。
- `clients`: 接入 Anthropic / OpenAI，并支持复用本机 Claude Code 配置。
- `tests`: 用假的模型响应验证循环行为，不依赖真实 API。

## 使用

```bash
uv sync
uv run python -m unittest discover
```

一次性调用：

```bash
uv run mini-agent "hello"
```

交互模式：

```bash
uv run mini-agent
```

默认 `--provider auto` 会优先复用本机 Claude Code 的 `~/.claude` 配置，其次使用 OpenAI 环境变量，最后回退到 `echo` 模型。

也可以显式选择：

```bash
uv run mini-agent --provider anthropic
uv run mini-agent --provider openai --model gpt-5.4
uv run mini-agent --provider echo
```

交互模式内置命令：

```text
/help
/model
/history
/clear
/exit
```

如果本机还没安装 `uv`，先安装 uv 后再运行上面的命令。

## 开源协议

本项目采用 MIT 协议开源。使用、复制、修改或分发本项目代码时，请保留 `LICENSE` 中的版权与许可声明。

如果你想在文章、课程、视频、仓库或项目文档中引用本项目，可以使用下面的信息：

- 项目: `learn-claude-code`
- 作者: `xielixing`
- 仓库: https://github.com/xielixing/learn-claude-code

仓库中也提供了 `CITATION.cff`，GitHub 会据此展示引用信息。

## 参考与致谢

这个仓库是个人学习用例，参考了墨角的《Claude Code 源码解读》教程：

- 教程首页: https://mo-jiao.github.io/claude-code-source-code/guide/

本仓库不会复制 Claude Code 源码；目标是用 Python 从零实现一个便于学习的最小 agent loop，并逐步补齐工具、配置、交互等机制。

## 下一步可逐渐补

- 增加 `read_file`、`write_file`、`bash` 等工具。
- 加入 SDK/JSON 输出模式。
- 增加上下文压缩、权限确认、token 用量统计。
