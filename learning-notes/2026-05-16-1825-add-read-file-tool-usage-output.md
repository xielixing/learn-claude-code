# 接入读文件工具并输出每轮 token 用量

## 解决了什么问题

本次任务希望 mini agent 更像一个真正的 agent loop：

- 支持模型多轮调用工具。
- 至少提供一个 `read_file` 读文件工具。
- 当模型不再返回工具调用时，自动停止循环。
- 控制台可以看到每轮模型调用的 token 用量。

## 解决之前怎么样

修改前，`QueryEngine.query_loop()` 里已经有多轮循环和工具结果回填逻辑，`tools.py` 里也已经有 `read_file_tool()`。

但从命令行启动时，`QueryEngine` 没有注册默认工具，所以真实模型拿不到 `read_file` 的工具 schema。用户输入类似：

```bash
uv run mini-agent --provider openai "请读取 README.md 并总结"
```

模型无法真正通过工具读取文件，只能根据已有上下文回答。

另外，`engine.py` 里的 assistant event 已经保存了 `usage`，但 `print_events()` 没有打印它，所以控制台看不到每轮 token 用量。

## 修改了哪里

- `mini_agent/cli.py`
  - 引入 `ToolRegistry` 和 `read_file_tool`。
  - 新增 `build_tools()`，把 `read_file_tool(".")` 注册成 CLI 默认工具。
  - 创建 `QueryEngine` 时传入默认工具集。
  - 新增 `format_usage()`，把 usage 字典格式化成控制台可读文本。
  - 在 `print_events()` 处理每轮 assistant event 后打印 `[usage] ...`。

- `tests/`
  - 根据当前学习目标，删除了原有测试用例文件。后续验证主要通过手动命令和小型运行片段完成。

- `AGENTS.md`
  - 新增项目协作规则，说明本项目默认是学习陪练模式，除非用户明确授权，否则 Codex 只给思路，不直接改源码。

- `learning-notes/TEMPLATE.md`
  - 新增稳定的学习记录模板，后续真正的学习笔记才使用时间戳命名。

## 修改思路

核心流程是：

1. CLI 收到用户输入。
2. CLI 创建 `QueryEngine`。
3. `QueryEngine` 每轮把当前 messages 和 tools schema 发给模型。
4. 如果模型返回 `tool_use`，agent 执行对应工具，并把 `tool_result` 作为下一轮 user message 回填。
5. 如果模型没有返回 `tool_use`，说明模型已经完成回答，循环直接停止。
6. 每一轮 assistant event 都带有 usage，控制台打印出来，方便观察调用成本。

这次不需要重写 `engine.py`，因为多轮工具调用和自动停止逻辑已经在那里。真正缺的是把已有的 `read_file_tool()` 接到 CLI 默认路径，并把已有 usage 展示出来。

## 解决之后怎么样

修改后，再输入类似：

```bash
uv run mini-agent --provider openai "请读取 README.md 并总结"
```

模型可以看到 `read_file` 工具。理想情况下，控制台会出现类似：

```text
[tool] read_file {'path': 'README.md'}
[tool result] ...
[usage] input_tokens=... output_tokens=... total_tokens=...
最终总结内容
[usage] input_tokens=... output_tokens=... total_tokens=...
```

其中：

- `[tool]` 表示模型请求调用工具。
- `[tool result]` 表示本地执行工具后的结果。
- `[usage]` 表示这一轮模型调用消耗的 token。
- 最后一轮如果没有新的工具调用，agent loop 会自动停止。

## 验证方式

因为测试用例已按要求删除，本次只做了小型手动验证：

```bash
uv run python -
```

执行一个内联脚本调用 `print_events()`，传入带 usage 的 assistant event，确认输出为：

```text
hello
[usage] input_tokens=3 output_tokens=2 total_tokens=5
```

后续如果要验证完整工具链，可以使用真实模型运行：

```bash
uv run mini-agent --provider openai "请读取 README.md 并总结"
```

## 本次经验

- agent loop 的关键不是一次请求模型，而是“模型回复 -> 工具执行 -> 工具结果回填 -> 再请求模型”的循环。
- 判断循环是否结束，不需要额外命令；只要模型不再返回 `tool_use`，就可以停止。
- 工具能力分两层：`tools.py` 定义工具本身，CLI 或 engine 初始化时要把工具注册进去，否则模型看不到工具。
- token usage 通常已经在模型响应里，功能上只差一个展示层。
- 学习型项目里，记录“修改前输入什么、修改后看到什么”比只记录代码改动更有价值。
