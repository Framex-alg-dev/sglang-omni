# Realtime Session 调试页面

该页面用于按 `session_id` 回放 Realtime 会话的所有 Turn，核对回复、动作 Prompt 和客户端关心的关键耗时。页面只读取本服务的结构化 JSONL 日志，不会向模型发起请求，也不会修改会话状态。

## 访问方式

服务启动后访问：

```text
http://<host>:<port>/debug/realtime
```

GPU 5 服务当前地址为：

```text
http://<host>:18002/debug/realtime
```

在页面输入 Session ID 后查询。也可以用 URL 参数直接打开：

```text
http://<host>:18002/debug/realtime?session_id=sess_xxxxxxxx
```

如果服务配置了 `SGLANG_OMNI_ADMIN_KEY`，需要在页面的 `Admin Bearer Token` 输入框中填写密钥。密钥仅保存在当前页面内存中，不写入 URL 或浏览器持久存储。

## 页面展示内容

每个 Turn 展示：

- Turn ID、触发来源、开始/提交时间和终态。
- 完整回复文本及回复来源（模型生成或客户端预置）。
- 回复 Prompt：有效 System Prompt、Messages 及已选动作类别等动态上下文。
- Category Prompt 和 Child Prompt：System Prompt、动态 Prompt、Messages、数字人状态、模型最终渲染 Prompt 及 PPL Top 12。
- 最终 Category、Candidate、`action_id`、支持状态和是否执行。
- Category/Child 计算耗时，以及以下两个端到端指标。

### 回复首包耗时

`reply.first_delta_after_commit_ms`：从服务收到合法 `turn.commit` 到发出第一个回复文本 Delta 的耗时。

客户端在 Turn 中直接提供回复文本时，该值通常接近协议处理耗时；这类 Turn 没有调用回复模型，页面会明确标记“回复 Prompt 未执行”。

### 最终动作耗时

`action.ready_after_commit_ms`：从服务收到合法 `turn.commit` 到发出 `turn.action.ready` 的耗时。这是客户端拿到最终动作并可以推进下一阶段的端到端指标，不等于单独的 Category 或 Child 模型计算时间。

## 数据接口

页面使用以下只读接口：

```http
GET /debug/realtime/api/session/{session_id}
Authorization: Bearer <admin-key>
```

如果服务未配置 Admin Key，`Authorization` 可省略。找不到 Session 时返回 HTTP 404，Session ID 格式非法时返回 HTTP 400。

## 日志与安全注意事项

- 页面从 `SGLANG_OMNI_REALTIME_LOG_DIR` 读取日志，因此只能查询尚未被清理的 Session。
- 要审核完整回复 instructions，需要启用 `SGLANG_OMNI_REALTIME_LOG_FULL_INSTRUCTIONS=1`。未启用时页面仅能显示非敏感的摘要和缺失标记。
- Prompt 可能包含用户输入、图片描述、人设和对话历史。生产环境应配置 Admin Key，并在网络层限制该页面的访问范围。
- 查询会扫描当前日志目录中的相关 JSONL 文件。它不会阻塞模型计算，但大量长期日志会增加查询耗时，应保持现有的按小时分目录和文件轮换策略。
