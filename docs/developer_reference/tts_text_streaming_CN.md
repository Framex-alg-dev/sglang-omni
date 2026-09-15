> 最新实现：默认启用有限合并缓冲，取代下文旧版本的逐delta透传默认；仅TTS输入受影响，客户端正文不变。
>
> `SGLANG_OMNI_TTS_TEXT_BUFFER_ENABLED=1`（默认）启用，设为0回到原发送策略。启用时优先于旧COALESCE设置，避免两层缓冲。首段60ms、后续100ms；最早未发送文本到达时开始计时，不因新delta重置。已有排队时间算入预算，发送背压不包含在此主动等待保证中。
>
> 长度按Unicode字母/数字计数（汉字也计入），首段16、后续40为软目标。句末边界最少4/8个有效字符；短回复在EOF立即排空。中文连续文本可在目标长度的汉字边界发送；英文优先空白边界，不新增词内切口。句末/分句符号随右引号和连续结束标点组合；英文句点因小数、缩写和URL歧义不单独触发发送，不使用词表。无法确认边界时由计时器发送已有内容，因此仍可能出现未完成短语。
>
> `buffer_deadline` / `punctuation` / `length` / `eof`记录发送原因。定时器独立等待队列，取消时释放等待任务。不能保证append就是最终合成段，也不保证消除下游音频供给不足。尚未重启服务或测定真实网关的首包收益。

# Embedded TTS 文本处理（仅 sglang-omni）

模型正文继续原样发送为 `response.text.delta`。TTS producer 为每次 `synthesize_streaming` 创建独立的空白处理状态，将处理结果按原有 `input_text_buffer.append` / `commit` 协议发给网关。没有网关、TTS模型或D_video_call改动。

## 默认行为

- 去掉整轮开头和结尾的普通空白。
- 连续普通空格和tab归一为一个空格；CR/LF/CRLF及相邻空白归一为一个换行。
- 保留词间空格、标点、缩写、重复正文及其他Unicode字符。
- 尾部空白等待下一有效字符确认；EOF丢弃剩余尾部空白。正文中间的空格不会因逐delta strip而丢失。
- 首个有效片段立即入队，不等单词/句子结束，不增加主动定时等待。普通append不是合成段，网关仍负责累计及最终分段。
- 默认不开启合并和大小切分：这两个功能只作为独立实验选项，不预设比透传更快。
- 全空白结果沿用无有效TTS文本的失败语义，不新增口头确认或改变沉默路由。

示例：`"  What's" + "  new?\r" + "\n  Hello " + "world.  "` 的TTS正文为 `What's new?\nHello world.`，客户端仍收到原文。

## 配置

生产launcher与开发CLI都读取以下变量；直接构造 `EmbeddedTTSConfig` 时使用显式字段，避免测试依赖外部环境。布尔值接受0/1、false/true，无效值在启动时失败。

| 环境变量（前缀 SGLANG_OMNI_TTS_TEXT_） | 配置字段 | 默认 |
|---|---|---:|
| NORMALIZE_WHITESPACE | normalize_text_whitespace | 1 |
| BUFFER_ENABLED | text_buffer_enabled | 1 |
| COALESCE | text_coalesce | 0 |
| APPEND_TARGET_CHARS | text_append_target_chars | 0 |
| MAX_TURN_CHARS | max_turn_text_chars | 16384 |
| MAX_TURN_BYTES | max_turn_text_bytes | 65536 |
| LOG_PAYLOADS | log_text_payloads | 0 |

`APPEND_TARGET_CHARS=0`不主动按大小切分。非零时优先选择已有空白或中文停顿标点，保留ASCII词、URL、撇号和组合字符；不可分内容允许超过软目标，但不能超过硬预算。它不是合成段最大长度。

`COALESCE=1`仅合并已经排队的文本，不sleep、不等下一delta；候选批大小采用APPEND_TARGET_CHARS，后者为0时采用512字符。遇到可能触发网关分段的标点优先发送。该选项不能保证减少最终合成段数。

原透传对照设置BUFFER_ENABLED=0、NORMALIZE_WHITESPACE=0、COALESCE=0、APPEND_TARGET_CHARS=0。资源预算和发送日志仍保留。

## 预算及隔离

producer在归一化前累计每轮原始字符和UTF-8字节，超出任一上限抛出`EmbeddedTTSError(phase="text_budget")`，不截断后冒称成功。reply层还会在入队前拒绝单次超大块。原有上游和producer队列均有数量上限；一个队列项受文本预算限制，因此排队内存有界。

预算针对整轮原始输入，而不是仅待发文本；长时间输出空白也计入。网关另有累计缓冲和段数限制，适配器不声称能从append大小推算网关剩余容量。

归一化器只属于当前producer任务。正常EOF、失败、取消、连接复用、预生成promote/discard沿用现有任务所有权与清理机制，不共享文本尾部。每轮声音instruct保持不变；超时已按下文“进展超时”更新。

## 日志

- `tts_text_append_sent`：实际成功发送的append，包含seq、字符/UTF-8字节数、源delta序号范围、触发原因、SHA256、缓冲等待、发送耗时和队列深度。
- `tts_text_normalization_completed`：输入/输出字符、输入字节、源delta数、去掉的前后空白、合并空白数、CR数量。
- LOG_PAYLOADS开启时，记录实际append.text及 `tts_text_delta_received` 原始TTS输入块。默认只记录元数据/hash。

源序号范围用于定位贡献该块的输入delta；切片可能共享同一源范围，不是精确字符偏移。`buffer_wait_ms`从TTS producer观察输入时开始，包含自身排队和instruct等待，不包含模型生成前耗时或上游队列已经发生的等待。不要拿它冒充全链路首包。

## 验证工具

```bash
PYTHONPATH=. .venv/bin/python benchmarks/eval/benchmark_embedded_tts_text.py \
  --voice spk_691b97a24dcc \
  --output reports/tts_text_validation_new \
  --repeats 20 --concurrency 1 2 --log-text
```

该工具直接运行本仓库EmbeddedTTSConnection，调用真实网关；不重启服务，不经过模型推理。对照raw、normalized、batched三组，轮换顺序，使用固定中文、英文缩写及极短回复，固定delta到达间隔，默认轮间等待3秒。`batched`合并已排队文本并启用512字符软目标；短文本不足以单独评估超长块切分收益，超长块正确性另有单元测试。

输出逐轮JSON、音频WAV及组统计。计时起点是适配器调用，不是API收到turn.commit；不得与原506轮模型端到端结果直接对比。首轮包含连接建立，后续复用连接；分析时应区分。组内混合不同用例，单用例样本不足时不能声称有稳定P95。

`supply_deficit_ms`是假设收到首包就开始播放的音频供给不足估计，不是实际浏览器卡顿。音频保存用于听审；自动化字符检查不等于音质验收。

完整集成测试覆盖原始text.delta不变、TTS文本归一化、连接复用、预生成、取消、失败、声音和session隔离。真实网关回放用于检查新适配器协议与供给行为；不宣称已验证完整真实模型多模态性能提升。

## 进展超时（后续修正）

原固定30秒整轮限制已调整。生产与开发入口使用同一 EmbeddedTTSConfig 默认值：

| 阶段 | 默认期限 | 刷新规则 |
|---|---:|---|
| 首包 | 10秒 | 首次非空文本append成功后开始；控制消息和空音频不刷新 |
| 音频停滞 | 10秒 | 每次有效非空音频交付后刷新；不受append或控制消息影响 |
| 结束确认 | 5秒 | 收到response.audio.done后等待response.done；不刷新 |
| 整轮硬上限 | 300秒 | 从本轮取得连接使用权开始，包含连接、文本生成等待、发送和接收；始终不刷新 |
| 音频交付背压 | 10秒 | 单次audio_sink调用，使用send_timeout_seconds；与网关音频停滞区分 |

协议不标注“最后一个音频delta”，所以audio.done之前只能使用停滞期限，不能从文本commit推断音频已经合成完毕。文本仍未产生下一段时，也可能触发停滞保护；这不是只测TTS引擎推理的计时器。

新错误阶段为audio_idle_timeout、completion_timeout、audio_delivery_timeout；首包和整轮仍使用first_audio_timeout、turn_timeout。tts_turn_started记录本轮的首包、停滞、结束和硬上限参数。每轮文本字符/字节预算及32MiB音频预算继续独立生效；未采用未经实测的“字符数→秒数”公式。持续输出也不会绕过300秒硬上限。

超时沿用取消和连接丢弃路径；取消清理耗时另计，错误送达时间可能晚于超时触发时间。本次没有修改网关的取消处理。
