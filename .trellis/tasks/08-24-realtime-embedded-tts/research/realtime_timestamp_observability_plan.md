# Session Realtime 关键时间戳观测方案（待审批）

## 1. 目标与边界

本文先定义 `/v1/session/realtime` 从服务启动、WebSocket、Session、Turn、模型、Action 到内嵌
TTS 的关键时间边界，供实现前审批。目标是回答以下问题：

- 服务和底层模型何时真正 ready；
- WebSocket、Session 和 Turn 初始化分别耗时多久；
- `turn.commit` 后，时间消耗在输入准备、模型排队、模型首字、TTS 首包还是外发；
- 中断从收到 `turn.cancel` 到模型、TTS 和外部终态收敛分别耗时多久；
- 延迟异常是否与日志丢弃、队列拥塞或资源状态有关。

本阶段不记录正文、Base64、PCM、图像、完整 URL/query、凭据、Cookie 或请求头。默认也不逐
Token、逐音频帧落一条日志；高频细节只在显式诊断模式中启用。

## 2. 当前代码的事实与需要纠正的口径

### 2.1 GPU/模型 ready 是服务级事件，不是 Session 级事件

生产路径在监听 API 前执行：

```text
MultiProcessPipelineRunner.start()
  -> 创建/启动 Stage 进程
  -> 分配 GPU、加载权重、建立 Pipeline
  -> 得到 coordinator
  -> 创建 Client
  -> 创建 FastAPI app 并开始监听
```

因此不存在“`session.start` 到达后再为该 Session 分配 GPU”的正常路径。正确拆分是：

- `model_pipeline_ready`：服务级，表示 GPU、权重和 Pipeline 已能接受请求；
- `session_resources_ready`：Session 级，表示 outputs、Action catalog/prefix、TTS manager 等本
  Session 资源准备完成。TTS manager 此时尚未联网；首个 audio Turn 才连接 provider。

fake model 路径不创建 GPU/Pipeline，应记录 `model_backend_ready backend=fake`，不能伪装为真实
模型加载完成。

### 2.2 绝对时间与耗时使用不同时间源

每条结构化事件建议同时捕获：

| 字段 | 时间源 | 用途 |
|---|---|---|
| `timestamp` | 带时区 ISO-8601，当前为微秒 | 人工查看、跨机器对齐 |
| `timestamp_unix_ns` | `time.time_ns()` | 同步时钟后的跨进程/跨服务关联 |
| `monotonic_ns` | `time.perf_counter_ns()` | 同一进程内精确计算耗时，避免系统时钟回拨 |
| `elapsed_ms` | 两个 monotonic 点计算 | 直接查询常用阶段耗时 |

当前 writer 已有 `timestamp` 和 `timestamp_unix_ms`。建议保留兼容字段并新增
`timestamp_unix_ns`、`monotonic_ns`，不修改既有字段含义。跨机器比较前必须确认宿主机使用
NTP/Chrony 同步；不同进程的 `monotonic_ns` 不能直接相减。

### 2.3 “WebSocket connect 前”需要客户端和服务端共同记录

服务端最早只能看到 ASGI 路由收到 upgrade，无法知道客户端调用 `connect()` 前的时间。从完整
系统的最终观测角度：

- smoke/数字人后端记录 `client_ws_connect_begin/client_ws_connect_end`；
- 本服务记录 `ws_upgrade_received/ws_accepted`；
- 双方未来可通过绝对时间、Session/Turn ID 和时钟同步结果离线关联。本次不新增
  `connection_trace_id`、握手 header 或 `session.start` 字段，只统计服务端区间。

## 3. 统一日志格式

复用 `sglang_omni.utils.structured_logs.emit_structured_log()`，继续输出 JSONL。公共字段：

| 字段 | 说明 |
|---|---|
| `schema_version` | 当前 `1.0`；新增时间字段时建议提升为 `1.1` |
| `timestamp` / `timestamp_unix_ms` | 保持现有兼容字段 |
| `timestamp_unix_ns` / `monotonic_ns` | 新增精确时间字段 |
| `log_type` / `event` / `level` | 现有分类和事件名 |
| `service_instance_id` / `hostname` / `pid` / `component` | 现有进程定位字段 |
| `session_id` / `turn_id` / `trace_id` | Session/Turn 关联字段 |
| `logical_request_id` / `request_id` / `response_id` | 模型和回复关联字段 |
| `backend` | `production` 或 `fake` |
| `outcome` / `error_phase` / `error_type` | 完成或失败摘要 |

所有时间在调用 `emit_structured_log()` 的业务边界立即捕获，然后进入异步队列；不能由 writer
线程在落盘时补时间，否则记录的是磁盘写入时间而不是事件发生时间。

## 4. 完整时间戳记录表

### 4.1 服务启动与模型就绪（service lifecycle）

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| S01 | `service_process_started` | CLI/launcher 进入服务启动 | backend、config hash | 进程启动基线 | 是 |
| S02 | `action_catalog_load_begin` | 生产 catalog 加载前 | catalog source 摘要 | catalog 加载耗时 | 是 |
| S03 | `action_catalog_load_end` | catalog 校验完成 | count、hash、elapsed_ms | 同上 | 是 |
| S04 | `pipeline_start_begin` | `MultiProcessPipelineRunner.start()` 前 | pipeline、stage count | 模型启动总耗时 | 是 |
| S05 | `model_stage_ready` | 每个关键 Stage 完成权重加载并可接收请求 | stage、rank、device、GPU ID、elapsed_ms | 各 Stage 加载耗时 | 是，低频 |
| S06 | `pipeline_start_end` | runner.start 返回 | pipeline、GPU count、elapsed_ms | Pipeline ready 耗时 | 是 |
| S07 | `model_pipeline_ready` | coordinator 可调用 | backend、pipeline、model | 模型服务 ready 时间 | 是 |
| S08 | `model_client_ready` | `Client(coordinator)` 完成 | backend | Client 构造耗时 | 是 |
| S09 | `api_app_ready` | FastAPI app、路由、manager 注册完成 | routes 摘要 | app 构造耗时 | 是 |
| S10 | `server_listening` | Uvicorn startup 完成并已绑定 host/port | 脱敏 bind address | 进程启动到可连接 | 是 |
| S11 | `service_shutdown_begin/end` | 优雅关闭边界 | reason、elapsed_ms | 清理耗时 | 是 |

说明：S05 需要在真正掌握权重/GPU ready 的 Stage 边界记录，不能只在 API 层根据
`runner.start()` 返回时间猜测。若底层暂时没有统一回调，MVP 至少实现 S04/S06/S07。

### 4.2 WebSocket 与 Session（connection/session lifecycle）

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| W01 | `client_ws_connect_begin` | 客户端调用 connect 前 | client trace、target label | 客户端建连耗时 | 客户端实现 |
| W02 | `ws_upgrade_received` | 服务端路由入口、`accept()` 前 | connection ID、peer 摘要 | 服务端 accept 排队 | 是 |
| W03 | `ws_accepted` | `await websocket.accept()` 返回后 | connection ID、elapsed_ms | upgrade→accept | 是 |
| W04 | `client_ws_connect_end` | 客户端 connect 返回 | client trace、elapsed_ms | 客户端端到端建连 | 客户端实现 |
| SS01 | `session_start_received` | 首条 `session.start` JSON 已解析、校验前 | session_id、outputs、protocol | connect→session start | 是，已有部分 |
| SS02 | `session_validation_completed` | outputs/字段/catalog 校验完成 | outputs、elapsed_ms | Session 校验耗时 | 是 |
| SS03 | `session_resources_ready` | Action prefix/prewarm、TTS manager 构造等结束 | backend、audio manager created、elapsed_ms | Session 资源准备耗时 | 是 |
| SS04 | `session_started_send_begin` | 外发 `session.started` 前 | session_id | 发送排队 | 可合并 |
| SS05 | `session_started_sent` | 外发完成 | outputs、elapsed_ms、model_pipeline_ready_unix_ns | session start→可用 | 是 |
| SS06 | `ws_disconnect_detected` | receive/send 检测断线 | state、reason category | 断线发生时间 | 是 |
| SS07 | `session_cleanup_begin/end` | 取消 Turn、关闭 TTS、release session 的前后 | cleanup flags、elapsed_ms | 断线清理耗时 | 是 |

### 4.3 Turn 输入和 commit（turn ingest）

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| T01 | `turn_start_received` | `turn.start` 已解析、业务处理前 | session/turn/trace、origin | Session idle→Turn | 是 |
| T02 | `turn_started_sent` | ACK 外发完成 | elapsed_ms | Turn 初始化耗时 | 是 |
| T03 | `turn_first_text_received` | 首次 text set/update | chars（不记正文） | 首输入等待 | 是 |
| T04 | `turn_first_audio_received` | 首个 audio append 校验完成 | seq、bytes、sample rate | 首音频到达 | 是 |
| T05 | `turn_first_image_received` | 首图像 append 校验完成 | seq、bytes/尺寸摘要 | 首图像到达 | 是 |
| T06 | `turn_input_summary_at_commit` | commit 时汇总 | first/last input mono time、chunks、bytes、frames、chars | 上传/采集窗口 | 是 |
| T07 | `turn_commit_received` | commit 业务处理入口 | 关联 ID、输入计数 | 所有 commit 派生指标基线 | 是，已有 |
| T08 | `turn_committed_sent` | `turn.committed` 外发完成 | elapsed_ms | commit ACK 延迟 | 是 |
| T09 | `image_preprocess_begin/end` | 图像预处理前后 | frame count、elapsed_ms | 图像准备耗时 | 有图像时 |
| T10 | `model_request_build_begin/end` | 构造 `GenerateRequest` 前后 | modalities、history count、elapsed_ms | 消息转换耗时 | 是 |

默认不逐 audio chunk 生成持久化事件。对象内保存 first/last 时间和计数，在 commit 时输出汇总；只有
诊断模式才逐 chunk 记录 `seq/bytes/queue_depth`，避免音频帧频率压垮日志。

### 4.4 模型回复（model/reply）

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| M01 | `model_stream_submit_begin` | 第一次拉取模型 stream 前 | request_id、backend | 请求准备→提交 | 是 |
| M02 | `model_stream_submit_accepted` | coordinator 接受/首 pull 建立后 | queue metadata（若可得） | 模型排队入口 | 是 |
| M03 | `model_first_chunk_received` | 任意首 chunk 到 API Client | modality、stage | coordinator TTFC | 是 |
| M04 | `model_first_text_delta_received` | 首个非空正文 Delta 到 Session | chars | commit→模型首字 TTFT | 是 |
| M05 | `response_first_text_delta_sent` | 首正文 Delta 外发完成 | response_id、chars | 模型首字→客户端外发 | 是 |
| M06 | `model_text_stream_completed` | 模型文本 EOF/finish | delta count、chars、finish reason、elapsed_ms | 模型流时长 | 是 |
| M07 | `response_text_done_sent` | text.done 外发完成 | response_id | commit→文本完成 | 是 |
| M08 | `model_abort_begin/end` | cancel/error 调用 Client.abort 前后 | request_id、success、elapsed_ms | 模型取消耗时 | 是 |

默认不逐文本 Delta 持久化，只记录首 Delta 与完成汇总。可选 `diagnostic_delta` 模式逐 Delta 记录
`seq/chars/queue_depth`，用于短时问题定位，不能作为生产默认。

### 4.5 TTS（embedded TTS）

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| V01 | `tts_turn_started` | audio 回复创建 TTS task | session/turn/response、reused candidate | TTS Turn 基线 | 是 |
| V02 | `tts_connect_begin` | 首次/重连 provider 前 | endpoint label（不可含 URL/query）、reason | TTS 建连耗时 | 是 |
| V03 | `tts_session_ready` | 收到 provider `session.created` | reused=false、elapsed_ms | connect→ready | 是 |
| V04 | `tts_connection_reused` | 健康连接被本 Turn 复用 | connection generation | 复用率 | 是，低频 |
| V05 | `tts_first_text_queued` | 首个模型 Delta 入 TTS 队列 | chars、queue depth | 模型首字→TTS queue | 是 |
| V06 | `tts_first_append_sent` | 首个 append 发送完成 | chars、elapsed_ms | queue→provider | 是 |
| V07 | `tts_commit_sent` | 模型 EOF 后 commit 发送完成 | text chunks/chars、elapsed_ms | 文本完成→commit | 是 |
| V08 | `tts_response_created` | provider response.created | provider response ID 摘要 | provider 接受耗时 | 是 |
| V09 | `tts_first_audio_received` | 首个合法 PCM 到 adapter | bytes | append→首音频 | 是 |
| V10 | `response_first_audio_delta_sent` | 首 PCM 外发后；provisional 则晋升释放后 | seq、bytes、provisional buffered | commit→外部首音频 | 是 |
| V11 | `tts_playable_250ms_ready` | 累计 PCM 达 250ms | bytes、chunk count | 可播放延迟 | 是 |
| V12 | `tts_stream_completed` | provider response.done | chunks、bytes、audio_ms、elapsed_ms | TTS 总耗时/尾部 drain | 是 |
| V13 | `response_audio_done_sent` | 外部 audio.done 发送完成 | last seq | 首音频→音频完成 | 是 |
| V14 | `tts_cancel_begin/end` | 内部 response.cancel、close 前后 | close outcome、elapsed_ms | TTS 取消耗时 | 是 |
| V15 | `tts_turn_failed` | adapter 唯一失败 owner | phase、exception type、close code、计数 | 故障阶段分布 | 是，已有基础 |

默认不逐 PCM chunk 落日志；记录首块、250ms 阈值和完成汇总即可。可选诊断模式逐块只记录
`seq/bytes/queue_depth`，绝不记录音频内容。

### 4.6 Action、正式终态和取消

| 编号 | 建议事件 | 记录边界 | 必要字段 | 主要派生指标 | 默认 |
|---|---|---|---|---|---|
| A01 | `action_scoring_begin` | Action 评分提交前 | candidate/category count | Action 总耗时 | 启用 Action 时 |
| A02 | `action_category_ready` | category 决策产生 | support status、elapsed_ms | category 延迟 | 启用时 |
| A03 | `action_child_ready` | child/action 决策产生 | candidate ID、elapsed_ms | child 延迟 | 启用时 |
| A04 | `turn_action_ready_sent` | action ready 外发完成 | elapsed_ms | Action 对外延迟 | 启用时 |
| R01 | `response_done_sent` | response.done 外发完成 | outputs、response_id | 回复完成延迟 | 是 |
| R02 | `turn_result_sent` | turn.result 外发完成 | status、outputs、elapsed_ms | Turn 端到端耗时 | 是，已有部分 |
| C01 | `turn_cancel_received` | cancel 业务处理入口 | processing phase | 取消基线 | 是 |
| C02 | `model_abort_completed` | 所有模型 request abort 收敛 | count、failures、elapsed_ms | 取消分段 | 是 |
| C03 | `tts_cancel_completed` | TTS cancel/close 收敛 | elapsed_ms | 取消分段 | audio 时 |
| C04 | `turn_branches_drained` | branch/inference/image tasks 已收敛 | task counts、elapsed_ms | 清理耗时 | 是 |
| C05 | `turn_cancelled_sent` | 唯一取消终态外发完成 | elapsed_ms | cancel→客户端终态 | 是 |
| E01 | `turn_failed` | Turn 失败 owner | phase、type、partial outputs、elapsed_ms | 故障率/失败延迟 | 是 |

## 5. 建议直接产出的延迟指标

| 指标 | 起点 → 终点 |
|---|---|
| `service_model_ready_ms` | S01 → S07 |
| `service_listen_ready_ms` | S01 → S10 |
| `ws_accept_ms` | W02 → W03 |
| `session_start_ready_ms` | SS01 → SS05 |
| `turn_ingest_ms` | T01 → T07 |
| `commit_ack_ms` | T07 → T08 |
| `request_build_ms` | T10 begin → end |
| `model_ttft_ms` | M01 → M04 |
| `commit_to_first_text_ms` | T07 → M05 |
| `first_text_to_tts_append_ms` | M04 → V06 |
| `tts_first_audio_ms` | V06 → V09 |
| `commit_to_first_audio_ms` | T07 → V10 |
| `commit_to_playable_250ms` | T07 → V11 |
| `model_text_total_ms` | M01 → M06 |
| `tts_tail_ms` | M06/V07 → V12 |
| `response_total_ms` | T07 → R01 |
| `turn_total_ms` | T07 → R02 |
| `cancel_total_ms` | C01 → C05 |

建议事件保留原始时间点，同时在终态记录常用派生值。这样既方便在线检索，也能在口径调整时从
原始点重新计算。

## 6. 持久化与性能评价

### 6.1 当前实现

当前 `PartitionedJSONLWriter` 已具备：

- 业务线程调用 `queue.put_nowait()`，不等待磁盘；
- 每进程一个 daemon writer thread；
- 队列默认 8192，满时丢弃并计数，不反压实时链路；
- 按本地小时、log type、component、PID 分文件；
- 默认单文件 128 MiB 滚动；
- 进程退出时最多等待 5 秒关闭。

不足是 writer 对每条记录执行一次 `open("ab", buffering=0)` 和一次 `write()`，没有真正批量
写入。它不会阻塞 asyncio，但会增加系统调用、目录检查和存储压力。

### 6.2 推荐批量策略

MVP 推荐：

- 调用线程仍然逐事件、立即捕获时间并 `put_nowait()`；
- writer thread 每次阻塞取得第一条后，再非阻塞 drain：
  - 最多 `100` 条；或
  - 从第一条起最多等待 `100ms`；
  - 任一条件先满足即提交；
- 按 `(hour, log_type, component, pid, segment)` 分组，把同文件记录拼成一个 bytes block；
- 每组一次 open/write，或者缓存有限数量文件句柄并按小时/滚动关闭；
- ERROR/CANCEL/SHUTDOWN 可唤醒 writer 提前提交，但仍不在事件循环线程执行磁盘 I/O；
- 不对每批 `fsync()`；依赖操作系统页缓存。若未来要求审计级 durability，再增加独立
  `fsync_interval_ms`，不能默认逐条 fsync；
- 记录 `queue_high_watermark`、`dropped_records`、`batch_records`、`batch_bytes`、
  `write_errors`，用于证明日志系统本身没有成为瓶颈。

推荐默认值：

| 配置 | 默认值 | 说明 |
|---|---:|---|
| queue capacity | 8192 | 延续当前值 |
| batch max records | 100 | 限制单批序列化/写入时间 |
| batch max delay | 100ms | 正常日志可见性和吞吐折中 |
| max file size | 128 MiB | 延续当前值 |
| shutdown flush timeout | 5s | 延续当前上限 |

风险取舍：进程被 `kill -9`、宿主机掉电时可能损失队列中和 OS page cache 中的最后一小段日志；
这是不阻塞实时事件链路的代价。正常退出必须 drain 队列。错误事件若要求更强持久性，推荐同时
输出标准 error 日志到容器 stdout，由容器日志驱动采集，而不是在 asyncio 线程 `fsync()`。

## 7. 实施分层建议

1. 先扩充 writer 时间字段和批量写，验证丢弃/滚动/关闭语义；
2. 增加服务启动和模型 ready 边界；
3. 增加 WebSocket/Session/Turn 边界；
4. 增加模型、TTS、Action 首点与终点；
5. 增加客户端 smoke 的 connect begin/end；
6. 更新 debug 汇总器和使用文档；
7. 用 fake model/fake TTS 注入固定延迟，校验派生指标误差；
8. 压测日志关闭、默认日志、诊断日志三种模式，比较 p50/p95/p99。

## 8. 建议验收标准

- 默认模式无逐 Token/逐 PCM 持久化事件；
- 所有关键边界在调用线程捕获时间，writer 排队时间不污染事件时间；
- 同一 Turn 可用 session/turn/trace/request/response ID 串出完整时间线；
- fake 注入 30ms 模型 Delta、50ms TTS 首包等延迟时，派生指标误差不超过 5ms；
- 日志开启相对关闭时，目标压测的 Realtime p99 增幅不超过 2%；
- 队列满时业务链路不阻塞，明确增加 dropped counter 和告警；
- 正常退出 drain；异常退出允许有界丢失；
- 日志中不存在正文、媒体、完整 provider URL/query 或凭据。

## 9. 已审批决策（2026-08-26）

1. 默认只记录“关键边界+首点+终点+汇总”；逐文本 Delta、逐 PCM 块日志仅在显式诊断模式
   启用。
2. 批量落盘默认采用“最多 100 条或最多等待 100ms，任一先满足即提交”。
3. 性能验收采用“默认日志开启后，目标 Realtime 压测 p99 相对关闭日志时增幅不超过 2%”。
4. 本次只实现本服务自身的服务端时间线。本服务记录 `ws_upgrade_received → ws_accepted`，不新增
   客户端握手字段、`connection_trace_id` 或数字人后端日志代码。数字人后端后续独立建立时间线，
   联调时通过双方绝对时间、Session/Turn ID 和时钟同步结果离线分析真正端到端延迟。

以上决策已获用户确认；文档仍是实施计划，不表示运行代码已经完成。
