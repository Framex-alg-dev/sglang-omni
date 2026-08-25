# Backend Development Guidelines

These guides describe the current Python service, pipeline runtime, and router conventions. They are grounded in `sglang_omni/`, `sglang_omni_router/`, and their tests.

## Guides

| Guide | Scope |
|---|---|
| [Directory Structure](./directory-structure.md) | Package boundaries and placement of new code |
| [Error Handling](./error-handling.md) | Domain, HTTP, WebSocket, streaming, and process failures |
| [Logging](./logging-guidelines.md) | Logger setup, levels, context, and sensitive data |
| [Quality](./quality-guidelines.md) | Formatting, tests, async lifecycle, and review checks |
| [Realtime Development Model](./realtime-development-model.md) | 无 GPU 本地 Realtime 模型替身的环境变量、边界与测试合同 |

There is no database or persistence layer in this repository: runtime state is in memory and inter-stage data travels through queues, ZMQ/control messages, or relay implementations. Add database guidance only if a real persistence package is introduced.
