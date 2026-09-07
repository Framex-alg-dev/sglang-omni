# Realtime 外置知识接入指南

完整架构与职责边界见[设计文档](../design/realtime_external_knowledge_base_CN.md)。本页只描述
当前 `sglang-omni` 已实现的客户端协议和服务端配置。

播客主线稿件的规范化、分段、TTS 元信息和场景动作匹配由 `D_video_call` 在角色创建阶段
完成，主线播放不请求 `sglang-omni`。只有用户插话形成用户 Turn 后，才进入本页描述的
Reply/Action 与 Knowledge Prepare/Commit 链路；回答完成后由 `D_video_call` 恢复主线。

外置知识实现位于同级仓库 `knowledge-platform`：自研编排服务在
`modules/knowledge-service`，RAGFlow 源码在 `modules/ragflow-service`。下面 URL 中的
`knowledge-gateway` 是当前部署 DNS 兼容名，不是仓库或模块目录名。

## 启动配置

本地 systemd 开发服务使用 `deploy/realtime-knowledge.env`。先从示例创建实际配置，填写
Knowledge Service 的运行时 Token，再由 service unit 通过 `EnvironmentFile` 加载：

```bash
cp deploy/realtime-knowledge.env.example deploy/realtime-knowledge.env
```

实际配置文件已加入 `.gitignore`，不要提交真实 Token。修改配置后需要执行
`systemctl daemon-reload` 并重启对应的 `sglang-omni` 服务。

也可以在不使用 systemd 时直接通过进程环境或 CLI 配置：

```bash
export SGLANG_OMNI_REALTIME_KNOWLEDGE_TOKEN='<service-token>'
sglang-omni serve ... \
  --enable-realtime \
  --realtime-knowledge-url http://knowledge-gateway:8090 \
  --realtime-knowledge-turn-timeout-ms 1200 \
  --realtime-knowledge-max-concurrency 128 \
  --realtime-knowledge-max-evidence 4 \
  --realtime-knowledge-max-context-chars 6000
```

可用环境变量：

- `SGLANG_OMNI_REALTIME_KNOWLEDGE_ENABLED`；未设置时，有 URL 即启用；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_URL`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_TOKEN`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_DEFAULT_TENANT`，仅用于可信单租户部署；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_CONNECT_TIMEOUT_SECONDS`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_TURN_TIMEOUT_MS`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_CONCURRENCY`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_EVIDENCE`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_MAX_CONTEXT_CHARS`；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_SPECULATIVE_ENABLED`，默认关闭；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_COMMIT_RESERVE_MS`，默认 200ms；
- `SGLANG_OMNI_REALTIME_KNOWLEDGE_COMMIT_RECOVERY_TIMEOUT_MS`，默认 500ms。

`TURN_TIMEOUT_MS` 默认 1200ms，作为从客户端并发槽等待到 Knowledge Service HTTP 响应的
单一绝对截止时间。服务端请求 deadline 会根据槽位等待后的剩余时间动态计算，并预留 100ms
网络返回余量，避免排队和 HTTP 调用分别消耗一份完整超时预算。

开启推测执行后，`sglang-omni` 会让 Reply/History Router 与 Knowledge
`turns:prepare` 并行。语言回复提交准备结果并使用 Evidence；纯动作只取消本地等待，未提交的
准备记录不会改变实体状态，并由 Knowledge Service 的短 TTL 自动清理。Commit 已发出后不随
Turn 取消，下一次知识操作会等待其同步最新 `state_token`。

多租户环境中的 `x-tenant-id` 必须由认证代理写入，不能原样信任公网客户端请求头。

## Session 绑定

客户端只能选择业务平台预先发布的 `binding_id`：

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "live-session-42",
  "outputs": ["text", "action"],
  "locale": "zh-CN",
  "knowledge": {
    "binding_id": "live-room-10001",
    "binding_revision": 3,
    "required": true
  }
}
```

`required=true` 时，Binding 不存在、越权或 Gateway 不可用会使 Session 启动失败；为
`false` 时 Session 可启动，但知识状态为 `degraded`。成功事件包含：

```json
{
  "type": "session.started",
  "knowledge": {"status": "ready", "snapshot_id": "snap_..."}
}
```

客户端不得提交 Gateway URL、Dataset ID、Profile、API Key、SQL 或任意检索过滤器。

## Turn 实体提示

在客户端已经由可信业务上下文确定实体时，可以随 commit 提交受限 hint：

```json
{
  "type": "turn.commit",
  "turn_id": "turn-7",
  "knowledge": {
    "entity_hints": [
      {"type": "product", "external_id": "sku-8848", "display_name": "云朵衬衫"}
    ]
  }
}
```

hint 不是授权依据。Gateway 必须按当前租户和 Snapshot 的实体目录再次验证。后续“这件
还有货吗”由 Gateway state token 解析跨轮实体，`sglang-omni` 不包含商品规则。

当前用户问答知识链路要求本轮有 `input.text.set`。在 `D_video_call` 集成中，麦克风音频由
D_video_call 做流式 ASR，并在 commit 前发送最终文本；sglang-omni 不重复执行 ASR。主动
Turn 的固定稿件不执行问答检索，但 D_video_call 会发送下述稿件生命周期事件来确定会话内
当前知识实体。

## 预生成稿件与知识实体切换

当主动口述内容是预先生成且已在 Binding Snapshot 中绑定的稿件时，推荐在开始播放前发送
独立生命周期事件：

```json
{
  "type": "knowledge.script.event",
  "request_id": "script-event-1",
  "script": {
    "id": "script-yi-quan-introduction-v1",
    "version": 1,
    "checksum": "sha256:<64位十六进制>"
  },
  "event": "started"
}
```

服务返回同 `request_id` 的 `knowledge.script.event.ack` 后才开始播放。正常完成发送
`completed`，打断或失败发送 `interrupted`；三个事件都携带完全相同的不可变
script id/version/checksum。Gateway 根据 Snapshot 中的稿件映射更新当前 Knowledge Unit 和
实体；客户端不需要、也不能提交 Dataset ID。`started` 校验失败且 Binding 为 required 时
不得播放稿件，从而避免稿件版本与知识版本错配。

协议仍兼容由提供文本的主动 Turn 携带稿件元数据，但 D_video_call 的持久化播放状态机使用
独立事件，因为一次完整稿件会被拆成多个播放 segment，生命周期不应绑定到单个模型 Turn。

## 决策和降级

Gateway 返回 `SKIP / RETRIEVE / CLARIFY / DEGRADED`。Evidence 会被限制数量与长度、
转义结构字符，并作为当前 user message 中的低权限数据注入；不会进入 system instruction。
`CLARIFY` 让模型只问一个最小澄清问题；`DEGRADED` 和无 Evidence 时禁止编造未经确认的
外部业务事实。知识任务与 Action 评分并行，取消 Turn 或断开 Session 会取消未完成任务。
