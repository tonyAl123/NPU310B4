# 使用自己的检查点复现转换

## 前提

使用 x86_64 Linux / WSL2。项目没有在原生 Windows 上安装 Linux CANN。
宿主用于导出和 CPU 数值验证，CANN 容器用于离线编译，开发板运行是后续独立验证。
固定上游提交：`15209f299fa372cccbd2226114f61a2f69f39a5f`，来源：https://github.com/MST19711/qwen-atlas-slm 。

## 1. 获取工具链安装器

自行从其官方来源获取并核对你有权使用的对应版本，将文件放入 `workspace/installers/`：

- `pixi-x86_64-unknown-linux-musl.tar.gz`
- `Ascend-cann-toolkit_7.0.0_linux-x86_64.run`
- `Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm`

根据实际下载内容生成 `SHA256SUMS`。本代码仓库不分发这些安装器和已安装容器镜像；请同时遵守其条款。
历史环境使用 CANN7/310B 容器配置、固定 Pixi 锁文件。替换版本必须重新核验支持，不能只改记录里的版本号。

## 2. 在 Linux 中建立编译环境

从仓库根目录运行：

```bash
bash workspace/setup-linux.sh "$PWD/workspace"
```

脚本可能通过 sudo 安装 Ubuntu 包、从固定上游克隆项目，并构建 CANN 镜像。所有内容都在执行前可阅读。
验收成功后生成 `workspace/environment-status.json`。本仓库不附带“当前机器已就绪”的状态文件来冒充新机器验收。
默认 Linux 构建根为当前用户的 `~/ascend-model-build`。

## 3. 放置完整本地模型

例如将你实际获准使用的 Qwen3.5 2B 检查点放入 `workspace/models/my-qwen35/`。
保留 safetensors、config、tokenizer、模板与 processor 文件。配置器不下载模型，也不把架构相似当作同一个权重版本。

## 4. 刷新本次检查点与源文件身份

```bash
python tools/configure_request.py --profile qwen35-ctx1024-img640   --model-dir "$PWD/workspace/models/my-qwen35"   --checkpoint-source "实际下载或派生来源"   --upstream-revision "可得的不可变revision"
```

若 revision 不可得，可不传该参数，记录会保留未知状态。配置器会读取元数据、核对结构、哈希模型文件，并刷新源文件校验。
这不是绕过校验：当前模型与发布配置有结构差异时会拒绝，需要另外适配。
它还会使用新机器验收得到的实际容器 ID，而不是强制复用发布者机器的镜像 ID。

## 5. 启动选定配置

```bash
bash workspace/convert-qwen35-ctx1024-img640.sh
```

其他名字见 `workspace/profiles.json`；启动器要求先运行配置器。
Gemma3 和 Qwen3.5 的共享构建记录进入 `workspace/builds`。TinyLlama 把结果写入 Linux 构建根下的 `results`。
所有结果以实际日志和新报告为准，完整构建需要相应 RAM、磁盘和工具链支持。

## 已知复现范围

- 本次仅对发布副本做语法、路径与发布计划检查，未运行模型导出/ATC/NPU。
- 其他新输入/依赖/工具链必须重做数值门槛；不能引用已有历史报告充当自己的新测试。
- Qwen3.5 为固定单图规格；原检查点缺少原图像 processor 配置，配套预处理有明确适配范围。
- 两份 OM 的配套 tokenizer、接口、dtype、三轴位置与 S/C/KV 状态都必须保持一致。
