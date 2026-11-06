# Ascend OM Conversion Lab

Chinese beginner guide, source-specific conversion adapters and numerical validation for Ascend310B4.

本项目把本地 Qwen3、Gemma3、TinyLlama 与 Qwen3.5 检查点适配为 Ascend310B4 OM。
代码与教程放 GitHub；Hugging Face 按家族与格式独立发布，4 个 ONNX 仓库和 4 个 OM 仓库。**这是非官方、逐检查点验证的转换项目，不承诺任意模型直接转换，也没有已验证的通用板端运行器。**

## 从哪里开始

- [模型转换入门长文](docs/index.html)：约 1.5 万字，19 章，含 4 个离线交互实验。GitHub 文件页不直接执行 HTML，可下载用浏览器打开，或启用 Pages。
- [复现步骤](REPRODUCE.md)：准备编译环境、配置自己的检查点、再启动对应转换。
- [发布流程](PUBLISHING.md)：GitHub 与 Hugging Face 的职责、账号与许可复核、上传和验收。
- [模型规格清单](publication/model-catalog.json)：历史成功构建的规格与内容身份。
- [验证记录与板端观察](VALIDATION.md)：包含历史数值验证和 Orange Pi AIpro 8T 测试汇总。
- [上游许可与修改说明](NOTICE)：保留 qwen-atlas-slm 的 MIT 声明与 Transformers 引用来源。

## 模型仓库规划

这些是建议名称，不代表仓库已经创建。实际地址由发布者命名空间决定。

| 家族键 | 格式 | Hugging Face 推荐仓库名 | 规格 |
|---|---|---|---|
| qwen3 | OM | `Local-Qwen3-0.6B-Ascend310B4-OM` | ctx256, ctx728, ctx1024, ctx2048, ctx8192 |
| qwen3 | ONNX | `Local-Qwen3-0.6B-ONNX` | ctx256, ctx728, ctx1024, ctx2048, ctx8192 |
| gemma3 | OM | `Local-Gemma3-1B-Ascend310B4-OM` | ctx512, ctx1024 |
| gemma3 | ONNX | `Local-Gemma3-1B-ONNX` | ctx512, ctx1024 |
| tinyllama | OM | `Local-TinyLlama-1.1B-Ascend310B4-OM` | ctx1280 |
| tinyllama | ONNX | `Local-TinyLlama-1.1B-ONNX` | ctx1280 |
| qwen35 | OM | `Local-Qwen3.5-2B-Ascend310B4-OM` | ctx1024-img640, ctx2048-img768 |
| qwen35 | ONNX | `Local-Qwen3.5-2B-ONNX` | ctx1024-img640, ctx2048-img768 |

目前共 4 个家族、10 个规格、12 个 OM、12 个 ONNX 图，以及它们引用的外部权重文件。Qwen3.5 的两个规格各包含视觉与 decoder 两套图。
历史 OM 构建目标均为 Ascend310B4。

用户报告在 Orange Pi AIpro 8T 上测试时，大部分受测模型使用单 NPU，生成速度约 3 token/s，部分模型的预填充（prefill）耗时约 10–30 秒。测试汇总尚未对应到具体规格或运行格式。详见[板端观察](VALIDATION.md#orange-pi-aipro-8t-板端观察)。

## 工程布局

```text
docs/                    单文件长文、证据摘要与公开数值报告
workspace/conversion-tools/ 导出器、原生参考、数值门槛和任务编排
workspace/profiles.json  10 个来源特定的转换配置
workspace/convert-*.sh   对应的 Linux 启动器
tools/configure_request.py  为自己的模型重新生成哈希与请求
upstream-reference/      固定提交的少量源代码/锁文件参考，不含 Git 历史
publication/             规格清单、发布配置模板、修改记录
```

## 可复现性的准确边界

历史原始构建通过了相应的本地数值门槛和 ATC。本次发布副本参数化了主机路径、添加了请求配置器，并规范化公开报告的绝对路径。
**参数化发布版本尚未在新建 Linux 环境重新跑完整转换**；不能把它的静态检查描述为新的模型编译成功。
在自己的检查点上运行时，必须生成新的参考、验证和构建记录。某些旧配置含专门的审计/恢复流程，参见 REPRODUCE.md。

## 许可

代码/文档准备稿拟按 Apache-2.0 发布，第三方原有通知继续保留；发布者应在真正发布前确认该选择。
上游 https://github.com/MST19711/qwen-atlas-slm 的 MIT 原文保留在 [LICENSES/upstream-MIT.txt](LICENSES/upstream-MIT.txt)。
代码许可不覆盖任何模型权重、tokenizer、CANN 安装器、容器镜像或其他运行时。
模型来源和适用许可必须单独核对，特别是本地 Gemma 微调卡与其基座条款之间的差异。
