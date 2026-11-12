# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循语义化版本。

## [Unreleased]

### Added
- HTTP API 层（`multi-agent[api]`）：`multi-agent-api` 启动 FastAPI 服务，
  提供 `POST /tasks`、`GET /tasks`、`GET /tasks/{thread_id}`、`GET /approvals`、
  `GET /approvals/{thread_id}`、`POST /approvals/{thread_id}`
- 进程内任务表与后台执行（`TaskManager`，`--max-workers` 控制并发），
  任务在遇到高风险动作时停在 `awaiting_approval` 并返回审批载荷
- API 审批决策复用 `record_approval`，与 CLI 写同一条 JSONL 审计链路
- `MULTI_AGENT_API_HOST` / `MULTI_AGENT_API_PORT` / `MULTI_AGENT_API_MAX_WORKERS`
- 取消任务：`DELETE /tasks/{thread_id}`（软取消，立即返回 `202` + `canceled`；已结束任务 `409`）
- 任务表治理：`--task-timeout` / `--retention` / `--max-records`
  （`MULTI_AGENT_API_TASK_TIMEOUT` / `MULTI_AGENT_API_RETENTION` / `MULTI_AGENT_API_MAX_RECORDS`），
  执行超时与待审批过期收敛为 `canceled`，已结束记录按保留期与容量上限回收
- `multi-agent-api --log-level`：启动器接管日志级别（默认跟随 `MULTI_AGENT_LOG_LEVEL`）
- 节点边界取消：`run_task_full` / `resume_task` 新增 `should_cancel` 回调，命中后停止后续节点并返回
  `stop_reason="canceled"`；API 的 `DELETE`、超时、审批过期都会给执行器发取消信号
- `evals/`：真实模型评测入口（`python evals/run.py`），要点命中打分 + tokens/耗时报告 + `--strict` 退出码
- 评测判分可选 `--judge` / `--judge-model`：由模型判断要点覆盖（同义改写也算命中），
  解析失败自动退回关键词打分；报告标注 `scored_by`
- 任务表持久化：`--task-db` / `MULTI_AGENT_API_TASK_DB`（默认 `.multi_agent/tasks.sqlite`，`memory` 关闭）。
  重启后已结束任务照旧可查，中断的 `queued` / `running` 标 `failed`，`awaiting_approval` 保留并可
  直接批准（执行器用同一 registry 重建图从检查点继续）
- API 鉴权与限流：`--api-token` / `MULTI_AGENT_API_TOKEN`（Bearer，`/health`、`/docs`、`/openapi.json` 免鉴权）、
  `--rate-limit` / `MULTI_AGENT_API_RATE_LIMIT`（每 IP 每分钟，`429` + `Retry-After`），新增 `GET /health`

### Changed
- 核心图未改动：API 复用 `run_task_full` / `resume_task` / `Settings`
- API 新增状态 `canceled`；`DELETE`、超时、审批过期都落在该状态，迟到结果不会复活任务
- 日志级别统一校验：`MULTI_AGENT_LOG_LEVEL` 与 `--log-level` 非法时启动即报错（CLI 原先静默退回 WARNING），
  可选 `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`
- 默认团队 prompt 重写：researcher 允许通识推断并限 250 字、同一缺口只查一次；
  writer 一次成稿限 600 字、不写收尾套话；reviewer 判据收敛为「必答点缺失 / 事实冲突 / 答非所问」，
  驳回必交 writer、已给意见后默认通过。supervisor prompt 增加「不连任、按需选人、返工上限、
  可交付正文才算 FINISH、reason ≤30 字」
- `_final_answer` 跳过审查驳回文本，优先取最后一份草稿；触顶时标注「未产出可交付草稿」
- `_decide_json` 兜底改为 `raw_decode` + 合法值清单，非法路由带清单重试一次
- 路由硬护栏：`AgentSpec.role`（researcher/writer/reviewer/executor）+ `MULTI_AGENT_MAX_REJECTIONS`
  （默认 1）；同一专家不连任（失败重试除外），reviewer 驳回达上限后直接收尾
- README 重组：`快速开始（克隆到跑通）` 六步清单 + `现状与完成度`（自评约 78%）+ `上线前你还需要自己做的`；
  结构树补齐 `api/store.py`、`evals/`、`tests/test_evals.py`
- 图编排拆分：`context_window.py`（消息转文本 / token 计数 / 裁剪）、`routing.py`（路由 schema、提示词、
  确定性护栏、降级解析）、`constants.py`（共用常量）；`graph.py` 只剩编排与执行（956 → 约 700 行），
  原有私有函数仍从 `multi_agent.graph` 重导出，调用方与测试无需改动
- README 更正 `run_parallel` 的连接说明：实际按数据库路径复用同一条 SQLite 连接（不是每任务一条），
  并附 8 任务压测数据（文件后端 1.11x / memory 后端 0.76x），说明该串行化在真实负载下可忽略

### Fixed
- 取消竞态：`DELETE` 落在「已批准、resume 未开始」之间时，已批准的高风险动作仍会执行。
  现在 `_resume` 复查任务状态，且审批提交与状态变更在同一把锁内
- 审计缺口：`resume_task`（`--resume` / `/resume` / `POST /approvals`）过去不写审计，现统一在 resume 路径写入；
  没有挂起审批时抛 `ValueError`，不再起幽灵 run 烧 token
- CLI 退出码：`stop_reason="error"` 的任务现在返回 `1`（原先恒为 `0`）
- 非交互审批：已输入 `y` 但备注行读到 EOF 时不再被判为拒绝
- `approval_tools` 写了不存在的工具名时构建期直接报错（原先静默不包审批闸）
- `approver` 返回 bool 时审计不再抛 `AttributeError`
- 空的 `MULTI_AGENT_API_HOST=` 不再让 uvicorn 绑到 `0.0.0.0`；`--log-level` 现在同时控制 uvicorn 日志
- `tools/compat_probe.py` 支持 `.env` / `OPENAI_*`，缺配置时给出用法提示而非 traceback
- 非有限数值校验：`--task-timeout nan`、`--retention nan/inf`、`OPENAI_TIMEOUT=nan/0/-1`
  启动即报错（原先 NaN 会静默关掉超时与回收）
- 检查点连接按数据库路径复用（原先按线程拆分且永不关闭，长驻进程 fd 单调增长）
- 任务表满（活跃记录达到 `--max-records`）时新提交返回 `503`，不再无界增长
- 打包：补 `license` / `classifiers` / `[project.urls]`（wheel 带 `License-Expression: MIT`），
  CI 增加 packaging job（build → 干净 venv 安装 `[api]` wheel → 校验入口与版本）
- API 版本号改为读取安装元数据，避免 `pyproject.toml` 与 `/openapi.json` 双源
- 并发建图时的检查点连接创建串行化（多个 worker 同时执行 `PRAGMA journal_mode=WAL`
  曾报 `database is locked`）
- DeepSeek V4 thinking mode：`make_llm` 对 DeepSeek 模型默认关闭 thinking，
  修复带 tools 时第二轮 400（`reasoning_content must be passed back`）
- `evals/` 每次运行使用独立临时检查点库与笔记目录（跑完删除），thread_id 带随机前缀，
  不再污染 `.multi_agent/`、`notes/` 或复用旧 checkpoint
- 上下文裁剪曾拆散 tool_call 与 ToolMessage：`trim_messages` 兜底按条数切片时可能留下无前驱的
  `ToolMessage`，或留下响应已被裁掉的 `tool_calls`，OpenAI 兼容接口会直接 400。现由
  `_drop_orphan_tool_messages` 双向修补；兜底分支也从"固定取最后 N 条"改为按预算从尾部收缩，
  避免截完仍超预算
- 审计日志并发丢行：`run_parallel` 的多个任务同写一个 JSONL，Windows 上 append 非原子，
  实测 400 条只剩 151 条、其中 45 行坏 JSON。改为 `record_approval` 全程持进程内写锁
  （跨进程并发仍不保证，已在注释中声明）
- `read_note` 会把任意大小的笔记整份读进模型上下文。增加 `MAX_NOTE_CHARS`（20 万字符）上限，
  超限截断并提示全文大小
- `_parse_first_json` 只认第一个 `{`：模型在 JSON 前解释一句且句子里带花括号时（如
  「我这样理解 {这一步是路由}」），降级路由会直接解析失败并终止任务。改为逐个 `{` 起点尝试解析
- 审批标记未校验来源：`_approval_markers` 现在只认本轮真正挂了审批闸的工具名。原先某个工具
  把外部内容原样回显（`REQUIRES_APPROVAL::<任意工具名>::{}`）时，就能伪造一次审批请求，
  借人工点「通过」把任意工具名写进 `approved_tools`，后续调用不再经过闸门

## [0.4.0] - 2026-10-27

### Added
- 工具级审批：`AgentSpec.approval_tools`，未批准的工具只回标记不执行，批准后重跑
- 审批前 `preview`：把将要提交的产出附在审批载荷里
- 定价工具支持 `vat_rate` / `return_rate` / `ad_rate`

### Changed
- 审计日志新增 `type` 与 `tool` 字段

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
