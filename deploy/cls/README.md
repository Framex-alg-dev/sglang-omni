# Omni CLS deployment

依据：[全链路日志规范 v1.0，第 7、8 节](https://ecnoie7eb92s.feishu.cn/wiki/OfPRw9rcdieqlOkxuEvcRz3vnzf)。
三个服务分别写 `/var/log/framex/<service>/app.log`，各进同名主题：
`omni-event-worker`、`omni-reply-worker`、`omni-action-worker`。
现有 QueueHandler、JSON、100MB×10 轮转保持原实现。部署采集器，不在推理请求中调用 CLS SDK。

国内使用既有 `framex-cn` 日志集；海外独立使用 `ap-singapore` 的 `framex-global`，不得跨境汇集。
`env=shared`。GPU 自建服务器使用该 CLS 地域的 HTTPS 公网接入点。
所有主题保留期至多 15 天，配置当天日志量超过过去七天均值三倍的告警。

## 目标配置与云端操作

先确认地域、三个既有 TopicId、机器组和已有采集配置；不要创建重复规则或重复读取轮转文件。
`targets.json` 是部署目录中的非秘密配置，包含 `region`、`host_log_root` 和 `topics`。
`topics` 的键恰好是三个服务名，值为不同 TopicId。`host_log_root` 必须是宿主机实际挂载目录。
应用内路径仍是代码常量，不新增日志路径环境变量。

当前 10 号机的三个容器均将 `/var/log/framex` 绑定到宿主机 `/data/nvme1/sglang-omni-unified-b1e5ff16-20261124/logs/framex`；采集路径须使用这个宿主机目录。
09 号机的 Omni inference gateway 另写 `/var/log/framex/omni-reply-worker/app.log`，应采入同一个 Reply 主题。该机只部署 Reply 采集规则，生成与审计命令都加 `--service omni-reply-worker`；不要给不存在的角色建立采集规则。

```sh
python scripts/prepare_omni_cls.py --targets /private-deploy/targets.json > /private-deploy/collection-requests.json
```

输出为 [CreateConfig](https://cloud.tencent.com/document/api/614/58620) 请求体，**不会调用云 API**。
配置管理员应先检查现有配置，按需创建或修改，再把规则绑定到目标机器组。
使用 JSON 提取、UTF-8、`timestamp` 毫秒与时区解析、失败行保留在 `LogParseFailure`。
仅采 `app.log`，由 LogListener 跟踪轮转后的 inode；不要同时匹配 `app.log.*` 导致重复。
默认从部署后增量采集（Backtracking=0），需要历史回填时单独决定范围。

键值索引不分词并开启统计：request_id、root_session_id、turn_id、trace_id、service、env、version、level、event、logger、host、user_id、space_id、character_id、voice_id、agent_session_id、gateway_session_id、dh_session_id、rtc_channel_id、provider_request_id、idempotency_key、peer、route、error_type。
数值索引：status、duration_ms。全文分词：message、error_message。

## LogListener 与密钥

使用[腾讯云官方安装指南](https://cloud.tencent.com/document/product/614/122939)安装宿主机 LogListener，指定真实 CLS 地域及网络方式。
凭据只交给 LogListener 并使用其加密存储；部署私密文件须为 0600。不放入 Git、示例、镜像、服务业务环境或验收报告。
每个写入身份仅能上传到所属主题。配置与查询由独立管理/只读身份完成，不给 Worker 扩大权限。
不要采内部 replay/diagnostic、提示词、ASR/LLM 内容、媒体与签名 URL；上线前先关闭逐帧服务日志。

## 验收

先审计宿主机文件，输出仅有计数和固定错误码：

```sh
python scripts/prepare_omni_cls.py --audit-root /actual/host/framex --root-session <root_session_id>
```

该结果只证明本地样本，**不表示已上传 CLS**。还需用只读账号在三个主题跨主题查询：
确认字段独立解析、时间与 timestamp 相差不超过一秒、中文正常、request_id 精确命中、禁写内容检索为零。
再跑真实会话，按 observation_id 连接 Event 窗口和 SensorTask，按 task 的 request_id/root_session_id/turn_id 核对 Reply、Action 和 ACK；保存 CLS 返回的查询证据。
