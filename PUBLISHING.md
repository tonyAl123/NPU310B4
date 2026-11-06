# GitHub 与 Hugging Face 发布步骤

## 两个平台分别放什么

- GitHub：转换工具、固定版本/输入协议、教程文章、模型目录、数值报告、许可通知。不要把整个原工作区 git add。
- Hugging Face：每个模型家族分成两个独立仓库，分别发布 OM 和 ONNX；4 个家族共 8 个仓库。每个仓库按 context/图片规格分目录，并附自己的 tokenizer、接口、预处理和数值记录。
- 本地 OM 准备目录为 `huggingface/om/家族/`，ONNX 为 `huggingface/onnx/家族/`。每个目录对应一个独立的远程仓库。
- OM 仓库的 OM 位于规格目录根层；ONNX 仓库的图位于 `规格/onnx/`，Qwen3.5 再分 `vision/` 与 `decoder/`，保持外部权重的原相对目录和文件名。使用 ONNX 不必下载 OM 仓库。
- 不上传：原始 safetensors、unbound/自测/失败中间图及临时权重、CANN 安装器、容器存储、系统磁盘、账号凭证。最终 ONNX 引用的 `.weights.data` 属于必须发布的模型载荷。

## 第一步：阅读准备稿

完整本地准备包的根目录中，START_HERE.md 是入口，release-catalog.json 是清单。
github/ 是代码仓库候选内容；huggingface/om/ 与 huggingface/onnx/ 各有 4 份模型卡与小型配套文件。
OM、ONNX 和外部数据没有再次复制；.local/local-sources.json 明确列出它们现有位置，上传工具只读取这些允许的文件。
`.local/` 与真正的 release-config.json 只留在自己的电脑，不要加入任何远程仓库。

## 第二步：离线检查

从完整发布准备包根目录运行（Python 3.10+）：

```powershell
python tools/release.py plan
python tools/release.py check
python -m pip install -r tools/requirements-publish.txt
python tools/release.py verify
```

plan/check/verify 不登录、不建立远程仓库、不上传。verify 会读取 release-catalog.json 所列原有模型载荷、校验 SHA256，并逐图解析 ONNX 的外部数据引用与字节范围，不执行模型、不重新导出或编译，也不另外复制大文件。完整校验需先安装 requirements-publish.txt。
check 检查准备清单是否齐全；verify 另外重新解析实际图文件。缺少 ONNX 或其权重文件会阻止实际上传。所有最终图及外部权重已与对应历史成功构建的哈希匹配。
check 扫描公开小型文本中的主机路径、常见令牌格式、私钥和 SSH 口令，以及模型二进制是否误入 GitHub 目录。大 tokenizer 数据按来源哈希检查，自动扫描不等于穷尽人工审阅。

## 第三步：补全身份与许可

复制 release-config.example.json 为 release-config.json（schema_version=2），然后填入 GitHub/Hugging Face 用户名或组织名。每个家族在 models.家族.repositories.om 与 repositories.onnx 分别设置仓库名和可见性，不可共用同一仓库名。
默认 private=true，若正式公开请明确改成 false。

每个模型还需填写真实 checkpoint_source、可得 revision、license_id；只有确知来源时才填精确 base_model。
参考家族不是对权重来源的自动证明。在该家族的 OM 和 ONNX 两个准备目录分别加入对应完整 LICENSE，再完成来源和许可复核字段。
GitHub 代码/教程拟用 Apache-2.0，保留上游 MIT；发布前确认 code_license_reviewed 和 public_files_reviewed。

特别注意：本地 Gemma 微调卡写 Apache-2.0，但其 Gemma 基座有独立条款与通知要求。不得仅复制卡内标签就认定可按 Apache 分发。
Gemma 的完整适用条款、使用限制、必要 Notice 与修改说明都需保留；目录已经提供 NOTICE 和 MODIFICATIONS 草稿。
Qwen/TinyLlama 候选家族为 Apache-2.0，但实际派生来源仍需确认。可参考附带的 Apache 许可文本，不能在确认前把它自动当作当前模型的 LICENSE。
ONNX 的内嵌权重与外部 `.weights.data` 都属于模型派生物，适用同样的来源和许可要求；外部数据不是仅含计算图的代码。

## 第四步：在自己的终端登录

准备好 Git、GitHub CLI 和 Python 环境。发布工具依赖与模型编译环境分开：

```powershell
python -m venv .venv-publish
.venv-publish\Scripts\python.exe -m pip install -r tools/requirements-publish.txt
gh auth login
.venv-publish\Scripts\hf.exe auth login
```

令牌只输入官方终端登录提示，不写进 JSON、脚本或模型卡，不需要粘贴到聊天中。
Python 发布工具复用正常 HF 登录缓存；命令行不接收明文 token 参数。

## 第五步：先发布代码仓库

先确认自己的 Git 提交署名已设置。可以仅在代码候选目录设置：

```powershell
git -C github/ascend-om-conversion-lab init -b main
git -C github/ascend-om-conversion-lab config user.name "你希望公开的署名"
git -C github/ascend-om-conversion-lab config user.email "你选择公开的提交邮箱或GitHub noreply邮箱"
```

先查看计划，再实际上传：

```powershell
.venv-publish\Scripts\python.exe tools/release.py publish-github
.venv-publish\Scripts\python.exe tools/release.py publish-github --upload
```

只有第二条会创建并推送仓库。工具拒绝覆盖已有仓库或替换已有 remote；如果仓库已经存在，应人工审阅 git status/diff，再使用正常 git 提交与推送流程。

## 第六步：分别发布 ONNX 与 OM

```powershell
.venv-publish\Scripts\python.exe tools/release.py publish-hf --family qwen35 --format onnx
.venv-publish\Scripts\python.exe tools/release.py publish-hf --family qwen35 --format onnx --upload

.venv-publish\Scripts\python.exe tools/release.py publish-hf --family qwen35 --format om
.venv-publish\Scripts\python.exe tools/release.py publish-hf --family qwen35 --format om --upload
```

带 --upload 才会写入 HF。--format onnx 只发布 ONNX 仓库，--format om 只发布 OM 仓库；缺少 --format 会报错。其他 family 为 qwen3、gemma3、tinyllama。可以只发布其中一种格式。
实际上传只读取选定格式的元数据和文件允许清单，核对来源/许可、载荷 SHA256（ONNX 另核对引用完整性），再通过 HfApi 创建该格式仓库。工具拒绝把 OM 二进制上传到 ONNX 仓库，或把 ONNX 图/权重上传到 OM 仓库。无需复制大文件或预先配置 Git LFS 才能使用此 HTTP 上传方式。
仓库若已存在会拒绝覆盖；网络失败可能留下空仓库或部分上传缓存，先检查远端状态，再选择恢复方案，不要反复创建新公开仓库。

## 第七步：发布后验收

1. 打开 GitHub 与 HF 页面，确认可见性、仓库名、作者、许可和非官方来源说明正确。
2. 确认 GitHub 没有 OM/权重/凭证，Pages 文章能显示；CI 不应被写成 NPU 测试。
3. 分别确认 OM 与 ONNX 仓库内有预期规格；Qwen3.5 的 OM 仓库每规格有两份 OM，ONNX 仓库每规格有两份 ONNX 及外部数据。按 onnx-manifest.json 确认引用齐全，并确认两种格式的 tokenizer 与接口各自齐全。
4. 下载一个规格到新的目录，用 SHA256SUMS 核对大载荷。
5. 把正式 HF 地址补回 GitHub README / 模型目录，模型卡再链接正式代码仓库。
6. 如要承诺板端效果，另行执行并公开真实设备、运行时、内存、准确性和速度测试。

## HTML 文章在线阅读（可选）

在 GitHub 仓库 Settings → Pages，选择从 main 分支的 /docs 发布（以账号实际支持的选项为准）。
docs/index.html 不依赖网络脚本或字体。启用 Pages 是单独的网站发布动作，本准备阶段没有执行。

## 官方依据（核对日期：2026-11-03）

- GitHub 本地项目导入：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- GitHub CLI 创建仓库：https://cli.github.com/manual/gh_repo_create
- GitHub 文件大小：https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- HF 模型卡：https://huggingface.co/docs/hub/model-cards
- HF 登录与 CLI：https://huggingface.co/docs/huggingface_hub/guides/cli
- HF 文件与自定义提交：https://huggingface.co/docs/huggingface_hub/guides/upload
- Gemma 条款：https://ai.google.dev/gemma/terms
- Gemma 限制：https://ai.google.dev/gemma/prohibited_use_policy
