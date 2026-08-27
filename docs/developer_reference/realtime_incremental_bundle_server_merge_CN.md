# Realtime TTS 增量 Bundle 服务器合并指南

本文用于把本地已经合并并验证的 Realtime、内嵌 TTS 和动作路由代码，迁移到
`/data/luozhijie` 下已有的服务器仓库。服务器仓库可能包含为了创建 uv 环境、安装依赖
或适配运行环境而产生的本地修改，因此本流程采用“保存服务器现状，再合并增量代码”，
禁止直接覆盖服务器工作区。

## 1. 本次交付物

增量 Bundle 文件名：

```text
sglang-omni-realtime-tts-action-incremental.bundle
```

Bundle 的前置提交为：

```text
6919c65ca470f30050c1e7a9e02a02438271c5ea
```

Bundle 中的目标分支为：

```text
integration/luozhijie-realtime-tts-action-suffix
```

这是增量 Bundle，不是完整源码压缩包。服务器仓库必须已经包含前置提交
`6919c65`，但服务器当前工作区和当前分支可以在该提交之上包含自己的修改。

## 2. 安全边界

本流程遵循以下规则：

- `/data/luozhijie` 不存在时立即停止，不创建替代目录；
- 不执行 `git reset --hard`、`git clean` 或强制 checkout；
- 不删除或重新创建服务器现有 `.venv`；
- 不把 `.venv`、模型权重、缓存、日志、密钥或 `.env` 提交到 Git；
- Bundle 只导入 Git 对象，`git fetch` 本身不会修改工作区；
- 服务器原有修改先保存在独立备份分支；
- 新代码只在新的部署整合分支中合并；
- 验证失败时可以切回备份分支。

## 3. 确认目录和文件

登录服务器后执行：

```bash
DEPLOY_ROOT=/data/luozhijie
test -d "$DEPLOY_ROOT" || {
  echo "ERROR: $DEPLOY_ROOT 不存在，拒绝继续"
  exit 1
}

BUNDLE_FILE="$DEPLOY_ROOT/sglang-omni-realtime-tts-action-incremental.bundle"
test -f "$BUNDLE_FILE" || {
  echo "ERROR: 找不到增量 Bundle：$BUNDLE_FILE"
  exit 1
}
```

指定服务器上已经能够运行的仓库目录。下面以
`/data/luozhijie/sglang-omni` 为例；如果实际目录名不同，只修改这一行：

```bash
REPO_DIR=/data/luozhijie/sglang-omni
test -d "$REPO_DIR/.git" || {
  echo "ERROR: $REPO_DIR 不是 Git 工作区，拒绝继续"
  exit 1
}

cd "$REPO_DIR"
```

## 4. 检查服务器现状

记录当前分支、提交和修改：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git diff --cached --stat
```

确认服务器仓库包含增量 Bundle 要求的前置提交：

```bash
git cat-file -e 6919c65ca470f30050c1e7a9e02a02438271c5ea^{commit} || {
  echo "ERROR: 当前仓库缺少前置提交 6919c65，不能使用本增量 Bundle"
  exit 1
}
```

`.venv/`、uv 缓存、模型权重和日志通常是未跟踪或被忽略的运行时文件，它们不需要
进入 Git，也不会因为后续 `git fetch` 而被删除。需要重点检查的是
`pyproject.toml`、`uv.lock`、启动脚本或 Python 源码等已跟踪文件。

## 5. 保存服务器上的有效修改

为服务器当前状态建立唯一的备份分支：

```bash
BACKUP_BRANCH="server/runtime-setup-backup-$(date +%Y%m%d-%H%M%S)"
git switch -c "$BACKUP_BRANCH"
echo "backup branch: $BACKUP_BRANCH"
```

如果 `git status --short` 没有已跟踪文件或需要保留的新文件，可以跳过本节剩余的
`git add` 和 `git commit`。

如果存在需要保留的修改，逐个明确添加。例如：

```bash
git add -- pyproject.toml uv.lock
```

如有服务器专用启动脚本，也应明确写出文件名：

```bash
git add -- path/to/server_start_script.sh
```

不要使用未经检查的 `git add .`。提交前必须查看实际内容：

```bash
git diff --cached --name-status
git diff --cached
git diff --cached --check
```

确认暂存区不包含密钥、Token、密码、`.env`、权重、日志或虚拟环境后再提交：

```bash
git commit -m "chore: preserve server runtime setup"
```

记录备份提交：

```bash
git rev-parse HEAD
```

## 6. 校验增量 Bundle

必须在已有 Git 仓库内部执行 `git bundle verify`，否则 Git 会报告
`need a repository to verify a bundle`：

```bash
cd "$REPO_DIR"
git bundle verify "$BUNDLE_FILE"
git bundle list-heads "$BUNDLE_FILE"
```

预期 `verify` 成功，并且 `list-heads` 中出现：

```text
refs/heads/integration/luozhijie-realtime-tts-action-suffix
```

同时核对交付时提供的 SHA256：

```bash
sha256sum "$BUNDLE_FILE"
```

如果 SHA256 不一致，停止操作并重新上传文件。

## 7. 从 Bundle 导入新代码

把 Bundle 目标分支导入为服务器本地的远端式引用：

```bash
git fetch "$BUNDLE_FILE" \
  refs/heads/integration/luozhijie-realtime-tts-action-suffix:refs/remotes/bundle/realtime-tts-action
```

检查导入结果及双方差异：

```bash
git log --oneline --decorate -10 refs/remotes/bundle/realtime-tts-action
git log --oneline --left-right --graph \
  HEAD...refs/remotes/bundle/realtime-tts-action
git diff --stat HEAD...refs/remotes/bundle/realtime-tts-action
```

此时服务器工作区仍未被新代码修改。

## 8. 建立服务器部署整合分支并合并

确保当前仍处于保存过服务器运行修改的备份分支，然后创建新的部署分支：

```bash
DEPLOY_BRANCH="deploy/realtime-tts-action-$(date +%Y%m%d-%H%M%S)"
git switch -c "$DEPLOY_BRANCH"
echo "deploy branch: $DEPLOY_BRANCH"
```

开始本地合并：

```bash
git merge --no-ff refs/remotes/bundle/realtime-tts-action
```

如果没有冲突，Git 会直接生成合并提交。如果出现冲突：

```bash
git status --short
git diff --name-only --diff-filter=U
```

逐个检查冲突文件，服务器运行环境相关修改与新代码都需要保留。尤其注意：

- `pyproject.toml` 和 `uv.lock`：保留服务器已验证的依赖环境，同时确认新代码依赖未丢失；
- Realtime 启动脚本：保留服务器模型路径、端口和运行参数；
- `sglang_omni/serve/realtime/`：不能简单选择单边版本；
- TTS 地址：服务器真实地址通常是
  `ws://127.0.0.1:40001/api-ws/v1/realtime`；
- 生产环境不得启用 `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED`。

解决每个冲突后明确暂存并完成合并：

```bash
git add -- <已解决的文件>
git diff --cached --check
git commit
```

不要使用 `git checkout --theirs .`、`git checkout --ours .` 或
`git reset --hard` 批量覆盖。

## 9. 检查 uv 环境

先确认服务器现有环境仍在：

```bash
test -x .venv/bin/python || {
  echo "ERROR: 服务器现有 .venv/bin/python 不可用"
  exit 1
}

.venv/bin/python --version
uv --version
```

如果合并没有改变依赖声明，不要无故重建环境。如果需要根据已有锁文件补齐依赖，优先使用：

```bash
uv sync --frozen
```

`--frozen` 可以避免验证期间意外重写 `uv.lock`。如果服务器原有适配要求不能使用
`--frozen`，先检查 `git diff -- pyproject.toml uv.lock`，再决定如何同步。

## 10. 合并后验证

确认当前提交、分支和工作区：

```bash
git branch --show-current
git log --oneline --decorate -5
git status --short
```

运行不需要 GPU 模型的核心测试：

```bash
.venv/bin/python -m pytest -q \
  tests/unit_test/test_structured_logs.py \
  tests/unit_test/serve/test_realtime_debug.py \
  tests/unit_test/serve/test_embedded_tts.py \
  tests/unit_test/serve/test_realtime_embedded_tts.py \
  tests/unit_test/serve/test_realtime_dev_model.py
```

本地合并时该测试集的基准结果为：

```text
59 passed
```

检查真实 TTS：

```bash
curl --fail --show-error http://127.0.0.1:40001/health
```

生产启动前确认 fake model 没有被环境注入：

```bash
env | grep '^SGLANG_OMNI_DEV_FAKE_' && {
  echo "ERROR: 检测到 fake model 环境变量，生产启动前必须清理"
  exit 1
}
```

如果没有输出且退出码为 `1`，表示没有匹配到 fake-model 环境变量，这是预期结果。

## 11. 回滚

如果合并后验证失败，先停止当前服务，然后查看本节前面打印的备份分支名：

```bash
git branch --list 'server/runtime-setup-backup-*'
```

在工作区没有未保存修改时切回对应备份分支：

```bash
git status --short
git switch <实际的备份分支名>
```

切换分支不会删除 `.venv`。如果失败发生在依赖同步阶段，应结合服务器保存的
`pyproject.toml`、`uv.lock` 和原 `.venv` 状态单独恢复，不要使用清理整个仓库的方式回滚。

## 12. 验收记录

建议保存以下信息用于部署追踪：

```bash
git rev-parse HEAD
git show --no-patch --format='%H%n%P%n%s' HEAD
sha256sum "$BUNDLE_FILE"
.venv/bin/python --version
uv --version
```

同时记录：

- 服务器备份分支名；
- 部署整合分支名；
- 冲突文件及处理方式；
- 核心测试结果；
- 真实 TTS health 结果；
- 实际生产启动命令和模型路径。
