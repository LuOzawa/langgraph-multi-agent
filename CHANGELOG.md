# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循语义化版本。

## [0.3.0] - 2026-10-27

### Added
- 高风险专家人工审批：`AgentSpec.requires_approval` + `interrupt()` 审批节点
- 跨进程恢复：`pending_approval()` / `resume_task()` / CLI `--resume`、`/pending`、`/resume`
- 审批审计日志（JSONL，`MULTI_AGENT_AUDIT_LOG`）
- 上下文 token 预算裁剪（`MULTI_AGENT_CONTEXT_TOKENS`，保留首条任务）
- token 用量统计（`RunResult.usage`）
- 独立任务并行执行（`run_parallel`，每个任务独立 thread 与 SQLite 连接）
- 跨境电商示例团队与 `calc_landed_cost` 定价工具
- GitHub Actions CI（Python 3.10/3.11/3.12：ruff + mypy + pytest）
- `docs/architecture.md`、`docs/integration-guide.md`、`docs/relay-compatibility.md`
- `py.typed`、`LICENSE`（MIT）

### Changed
- 每个专家可配置 `max_steps`，覆盖全局 `MULTI_AGENT_WORKER_STEPS`
- 专家只看到业务消息，supervisor 路由记录不再进入专家上下文
- 专家产出保留原始 `AIMessage`（含 content blocks）

### Removed
- 无实际读取方的 `next_agent` 状态字段

## [0.2.0] - 2026-10-27

### Added
- SQLite 检查点与 `thread_id` 多轮会话
- 路由枚举由注册表动态生成，`structured output` 失败时退化为 JSON 解析
- 专家失败隔离、`max_turns` 保护、每轮日志

## [0.1.0] - 2026-10-27

### Added
- 初始骨架：Supervisor 调度 researcher / writer / reviewer
- 本地笔记工具、CLI 入口
