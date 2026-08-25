# Session Realtime 本地开发模型替身规范

## 1. 适用范围与触发条件

本规范用于在没有 GPU、模型权重和 Pipeline 子进程时验证唯一的 Session Realtime
protocol v1 接口 `WS /v1/session/realtime`。替身复用正式 `MultimodalSession` 的
协议解析、Session/Turn 状态机、输入 ACK、文本、动作、融合、取消和终态逻辑，只
替代模型文本生成与动作评分能力。

该能力仅用于开发，不是生产降级路径，不实现 TTS、回复 PCM、`response.audio.*`
或旧 `/v1/realtime` 兼容。开发 app 的外部 allowlist 必须精确为：

- `GET /health`；
- `GET /v1/models`；
- `WS /v1/session/realtime`。

独立入口是 `python -m sglang_omni.serve.realtime.dev_server`。正式 launcher 仅在
`SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED=true` 时进入相同开发分支；关闭时保持生产
`action catalog -> port -> Pipeline runner` 顺序。

## 2. 接口签名与模块边界

`MultimodalSession` 对替身 client 只使用窄接口，不继承需要 Coordinator 的生产
`Client`：

```python
class RealtimeModelClient(Protocol):
    def completion_stream(
        self, request: GenerateRequest, *, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]: ...

    async def score_action_suffixes(
        self, request: ActionSuffixScoreRequest
    ) -> ActionSuffixScoreResult: ...

    async def abort(self, request_id: str) -> AbortResult: ...
```

`DevRealtimeModelClient.__getattr__` 必须抛 `AttributeError`，确保共享代码的
`hasattr()` 能力探测符合 Python 语义。非 allowlist 路由应在传输边界拒绝，不能靠
缺失 client 方法产生偶然 500。

| 环境变量 | 合同 |
|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | 严格 `true`/`false`，默认 `false` |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 非空固定回复正文 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | 正整数，按 Unicode 字符切片 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | 非负整数 |
| `SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID` | 可空；非空时必须是 Session 白名单中的非保留 candidate |

## 3. 协议与行为合同

第一条消息必须是 `protocol_version=1` 的 `session.start`，之后使用正式
`turn.start`、`input.text.set`/`input.audio.append`/`input.image.append`、
`turn.commit`、`turn.cancel`、`session.close`；替身不得增加第二套事件解析器。

支持 `outputs=['text']`、`['action']`、`['text','action']`：

- text：固定正文通过正式 `response.created -> response.text.delta* ->
  response.text.done -> response.done -> turn.result` 输出，正文等于 delta 拼接；
- action：只从已验证的 `allowed_candidates` 选择；空配置确定性选择首个合法候选；
- text+action：走正式 provisional promotion/discard 与 action ready/result；
- audio：当前稳定返回 `unsupported_output`，不得静默降级为 text。

开发内联 action catalog 必须保留生产约束：拒绝 `A000`、`B000`、
`UNSUPPORTED`、内部 `DEV_ACTIONS`/`DEV_NONE_*` 冲突、重复 ID、category/candidate
交集、非法或超限 execution binding。配置 candidate 不在 child 评分白名单时明确
失败，不得用保留 ID 绕过 unsupported/fallback 状态机。

真正取消要求 `turn.cancel` 设置 cancelling，abort 活动 request，取消并等待
reply/action/image/inference task，只发送一次 `turn.cancelled`。取消开始后不得发送
`response.done`、`turn.action.ready` 或 `turn.result`。重复 cancel、完成与迟到
cancel 竞争、断线和重复 teardown 必须幂等；收到 `turn.cancelled` 后可开始新 Turn。

## 4. 校验与错误矩阵

| 条件 | 稳定行为 |
|---|---|
| 环境布尔、整数、正文或配置 candidate 非法 | uvicorn、catalog、Pipeline 启动前 `ValueError` |
| 第一条消息不是 v1 `session.start` | 正式协议错误，Session 不进入 started |
| outputs 含 `audio` 或未知值 | `error.code=unsupported_output` |
| action 缺少 fallback/allowed candidates | `session_candidate_invalid` |
| 保留/重复/冲突 ID 或非法 binding | `session_candidate_invalid` |
| turn ID、seq、PCM16LE 或图片非法 | 正式 `invalid_turn_id`、`invalid_sequence`、`invalid_audio`、`invalid_image` |
| 非 allowlist HTTP | HTTP 501 + `dev_fake_model_unsupported` |
| 旧 `/v1/realtime` 或其他 WebSocket | 不注册、不接受连接 |
| 活动 Turn cancel | 一次 `turn.cancelled`，无后续成功终态 |
| 重复或迟到 cancel | 保留第一个终态，不发第二终态或额外错误 |
| 断线/session.close | 幂等取消全部任务并释放 Session |

错误载荷和日志不得包含 Base64 媒体、完整固定回复、凭证或任意 traceback。

## 5. Good / Base / Bad Cases

- Good：无 GPU 启动独立入口，以 text+action 建立 v1 Session；提交 Turn 后收到
  provisional、固定文本、白名单 action 和唯一 `turn.result`；慢 Turn cancel 后只
  收到一次 `turn.cancelled`，随后新 Turn 正常开始。
- Base：`ENABLED=false` 时生产 launcher 仍先加载/校验 action catalog，再探测端口
  并构造 runner；resource monitor 和生产路由默认启用。
- Bad：自行解析一套 fake WebSocket 事件；暴露旧 `/v1/realtime`；接受保留 ID 或
  空白 binding；cancel 后继续发送成功终态；把 WAV header 当 raw PCM16LE；在本任务
  实现回复音频/TTS。

## 6. 必需测试与精确断言

- 配置：严格布尔/整数、空正文、保留 action ID、日志摘要不含正文；
- client text：Unicode chunk 拼接等于配置正文，独立 stop chunk，abort 后无 stop；
- client action：配置 candidate 在 child 白名单中胜出，不在白名单明确失败；
- Session：真实 `create_app()` + TestClient/WebSocket 覆盖 text、action、融合、
  audio unsupported、重复 Turn；
- action：逐项断言保留/内部/重复/冲突 ID 和非法 binding 被拒绝；
- 取消：慢流 commit 后 cancel，断言 `turn.cancelled` 恰好一次，且其后不存在
  `response.done`、`turn.action.ready`、`turn.result`；重复/迟到 cancel 无第二终态，
  新 Turn 可开始；
- 路由：WebSocket 集合严格等于 `{'/v1/session/realtime'}`；HTTP 仅 health/models
  成功，其余代表路由为 501；
- launcher：fake 的 catalog/runner 构造器为 forbidden；disabled 顺序严格为
  `catalog -> port -> runner`；
- smoke：检查 ACK、`turn.committed`、delta 聚合、done/result、action 一致性、
  cancel、乱序、timeout、服务 error、意外关闭和非零退出；WAV 必须是非空、未压缩、
  16 kHz、单声道、PCM16，并且只发送 raw frames。

## 7. Wrong vs Correct

### Wrong：在资源分配后分流，或复制协议

```python
runner = MultiProcessPipelineRunner(config)
if fake.enabled:
    return run_fake_websocket_parser()
```

### Correct：资源前分流并复用正式 Session

```python
fake = DevRealtimeModelConfig.from_env()
if fake.enabled:
    return await serve_dev_realtime_model(fake)  # create_app + MultimodalSession

catalog = load_production_action_catalog()
port = find_available_port(...)
runner = MultiProcessPipelineRunner(config)
```

### Wrong：cancel 只改 bookkeeping

```python
cancelled_ids.add(turn_id)
await send({"type": "turn.cancelled"})
```

### Correct：先终止 owner，再发送唯一终态

```python
turn.phase = "cancelling"
await abort_active_requests(turn)
await cancel_and_join_turn_tasks(turn)
clear_turn_resources(turn)
await send_once({"type": "turn.cancelled", "turn_id": turn.turn_id})
```
