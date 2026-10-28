# 中转 / 模型兼容性实测

测试脚本不落库：直接读本机 `~/.openclaw/openclaw.json` 的 provider 配置，用项目自带 `ChatOpenAI` 打三个探针，最后跑一次真实端到端任务。

## 探针结果

| provider | 模型 | 普通对话 | 工具调用 (tools) | structured output | 备注 |
| --- | --- | --- | --- | --- | --- |
| deepseek | deepseek-v4-flash | ✅ 2.01s（首轮）；后续两次探针出现 `OpenAIConnectionError` 7.2s | ✅ 1.07s（正确产出 `get_weather` 调用） | ❌ HTTP 400 `This response_format type is unavailable now` | 路由走 JSON 兜底 |
| zai | glm-5.3 | 未测 | 未测 | 未测 | 本机该 provider 无可用 apiKey |
| minimax | MiniMax-M3 | 不适用 | 不适用 | 不适用 | `anthropic-messages` 协议，不走 `ChatOpenAI` |

## 端到端（deepseek-v4-flash，真实调用）

```
任务：用一句话说明 LangGraph 里的 checkpoint 有什么用
结果：turns=2, stop_reason=finish, elapsed=6.39s
日志：structured output 不可用（400）→ 退化为 JSON 解析 ×2
答案：LangGraph 的 checkpoint 用于持久化保存图执行过程中的状态与进度，
      从而支持中断恢复、human-in-the-loop、时间旅行调试和容错重试。
```

> 网络备注：本机到 `api.deepseek.com` 的连通性不稳定（同一分钟内 `chat` 探针时通时断），但 tool calling 与端到端任务均成功。连接错误与模型能力无关，属于网络层问题；项目侧靠 `OPENAI_MAX_RETRIES` 重试缓解。

**结论**：deepseek 官方接口不支持 `response_format: json_schema`，但支持 tool calling 与普通 JSON 输出。项目的降级路径（先 structured output，失败后要求模型输出 JSON 再解析）在真实模型上验证有效，不是只能跑假模型。

## thinking mode（v4 系列）

DeepSeek V4 默认开启 thinking mode（`reasoning_content`）。官方文档要求：请求带 `tools` 时，
历史轮次的 `reasoning_content` 必须回传，否则 400
`The reasoning_content in the thinking mode must be passed back to the API`。
`langchain-openai` 目前不提取/透传该字段，所以工具循环第二轮必失败。

项目处理：`make_llm` 对 DeepSeek（base_url 或模型名包含 `deepseek`）默认加
`extra_body={"thinking": {"type": "disabled"}}`，工具链恢复正常；代价是没有 CoT 推理。
实测关闭 thinking 后 `evals/run.py` 的 tool-calling 用例可通过，不再出现该 400。

## 复现

```bash
# 读 .env / OPENAI_* 环境变量（推荐，见 README 本地开发第 3 步）
python tools/compat_probe.py
# 或读本机 openclaw provider 配置
python tools/compat_probe.py --provider deepseek
# 或显式传参
python tools/compat_probe.py --api-key sk-... --base-url https://... --model ...
```

## 尚未覆盖

- 商业中转（one-api / new-api 类）：未拿到可用密钥
- 只支持 `/v1/chat/completions` 前缀差异的网关：`OPENAI_BASE_URL` 会自动 `.rstrip("/")`，但不会补 `/v1`
- 流式与并发：本表只测非流式单请求
