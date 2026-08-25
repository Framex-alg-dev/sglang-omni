# Session Realtime 本地开发模型替身：技术设计

## 1. 边界

```text
生产：launcher -> action catalog -> Pipeline -> Client -> create_app -> MultimodalSession
开发：dev_server / launcher fake branch -> DevSessionModelClient -> create_app -> MultimodalSession
```

替身位于 `MultimodalSession` 与生产 Client/Pipeline 的边界。外部协议解析、Session/Turn 状态机、媒体 ACK、provisional、取消和终态继续使用正式实现；替身只提供确定性的模型与动作结果。

## 2. 路由与能力

开发 app 使用显式 allowlist：

- HTTP：`/health`、`/v1/models`；
- WebSocket：`/v1/session/realtime`。

共享 `create_app()` 后续新增路由不会自动进入开发模式。旧 `/v1/realtime` 必须移除。生产模式默认注册所有正式路由和 resource monitor。

输出组合由正式 Session 的统一能力解析器校验。本任务支持 text、action、text+action；audio 暂时返回明确错误。TTS 任务将扩展同一解析器，而不是在 fake client 中增加独立开关。

## 3. 窄 client

从 `MultimodalSession` 的实际调用点提取 Protocol：文本生成保留 `completion_stream()`，动作路径提供最小的固定 action scoring/结果能力，并让 `hasattr()` 探测保持标准 Python 语义。Protocol 不继承需要 Coordinator 的生产 `Client`。

固定动作只能从 Session 已验证白名单选择，并复用服务端全局目录解析 category/action 元数据。配置 candidate 非法时失败；不在替身中复制目录结构。

## 4. 数据流

```text
client session.start(outputs)
  -> 正式协议校验与 Session 建立
client turn.start + input.* + turn.commit
  -> MultimodalSession 冻结 Turn
  -> DevSessionModelClient 固定 text/action
  -> 正式 provisional/response/action/turn.result 事件
```

text+action 必须走正式 provisional promotion/discard 分支，从而为后续 audio 缓冲测试提供真实骨架。

## 5. 取消与清理

Turn owner 持有文本、动作和发送任务。`turn.cancel` 设置终态 token，取消任务并等待清理，然后发送一次 `turn.cancelled`。迟到结果在发送边界按 token 丢弃。`session.close`、断线和重复 teardown 都必须幂等。

## 6. 启动兼容

- 独立 dev server 不解析模型配置，不加载 action catalog、resource monitor 或 GPU/Pipeline 资源。
- launcher fake 分支在生产 action catalog 的懒导入之前返回。
- disabled 分支维持合并后的 `catalog -> port -> runner` 顺序。

## 7. 测试结构

- client 单测：配置、固定 chunk、固定 action、取消；
- Session 集成：真实 `create_app()` + TestClient/WebSocket，覆盖三种输出；
- 路由测试：开发 allowlist 精确等于目标集合；
- launcher 测试：fake 不加载 catalog/runner，disabled 顺序不变；
- 进程外 smoke：真实端口和协议事件；
- 所有测试禁止模型下载、GPU 和网络 provider。

## 8. 迁移与回滚

删除旧接口专用测试、smoke 事件和文档，不保留双协议兼容层。回滚开发替身只需关闭 fake 环境变量；生产 `/v1/session/realtime` 不受影响。
