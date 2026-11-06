# Ascend OM Conversion Lab

Chinese beginner guide, source-specific conversion adapters and numerical validation for Ascend310B4.

本项目将本地 Qwen3、Gemma3、TinyLlama 与 Qwen3.5 检查点导出为 ONNX，并编译为 Ascend310B4 OM。代码和教程在本仓库，模型产物按家族和格式分别上传到 Hugging Face。

## 从哪里开始

- [模型转换入门长文](docs/index.html)：约 1.5 万字，19 章，含 4 个离线交互实验。GitHub 文件页不直接执行 HTML，可下载用浏览器打开，或启用 Pages。
- [复现步骤](REPRODUCE.md)：准备编译环境、配置自己的检查点、再启动对应转换。
- [发布流程](PUBLISHING.md)：GitHub 与 Hugging Face 的职责、账号与许可复核、上传和验收。
- [模型规格清单](publication/model-catalog.json)：历史构建规格、源检查点和上传状态。
- [验证记录与板端观察](VALIDATION.md)：包含历史数值验证和 Orange Pi AIpro 8T 测试汇总。
- [上游许可与修改说明](NOTICE)：保留 qwen-atlas-slm 的 MIT 声明与 Transformers 引用来源。

## Hugging Face 模型仓库

以下六个仓库已上传并核对文件，目前均为私有仓库，需要有权限的 Hugging Face 账号才能访问。

| 模型 | OM | ONNX | 规格 |
|---|---|---|---|
| Gemma3 1B | [OM](https://huggingface.co/TonyAlexie/Local-Gemma3-1B-Ascend310B4-OM) | [ONNX](https://huggingface.co/TonyAlexie/Local-Gemma3-1B-ONNX) | ctx512, ctx1024 |
| TinyLlama 1.1B | [OM](https://huggingface.co/TonyAlexie/Local-TinyLlama-1.1B-Ascend310B4-OM) | [ONNX](https://huggingface.co/TonyAlexie/Local-TinyLlama-1.1B-ONNX) | ctx1280 |
| Qwen3.5 2B（AkaumiKaede 派生版） | [OM](https://huggingface.co/TonyAlexie/Local-Qwen3.5-2B-Ascend310B4-OM) | [ONNX](https://huggingface.co/TonyAlexie/Local-Qwen3.5-2B-ONNX) | ctx1024-img640, ctx2048-img768 |

Qwen3 0.6B 的五个规格（ctx256、ctx728、ctx1024、ctx2048、ctx8192）保留在本地，OM 与 ONNX 上传待确认派生权重的再分发许可。[源检查点](https://huggingface.co/DavidAU/Qwen3-0.6B-heretic-abliterated-uncensored)

历史构建覆盖 4 个模型家族和 10 个规格。Qwen3.5 的每个规格都包含视觉和 decoder 两套图，OM 的目标芯片均为 Ascend310B4。

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

本仓库代码和文档的许可见 [LICENSE](LICENSE)，第三方原有声明随仓库保留。
上游 https://github.com/MST19711/qwen-atlas-slm 的 MIT 原文保留在 [LICENSES/upstream-MIT.txt](LICENSES/upstream-MIT.txt)。
代码许可不覆盖任何模型权重、tokenizer、CANN 安装器、容器镜像或其他运行时。
模型检查点和适用许可记录在模型清单与各 Hugging Face 模型卡中。Gemma3 仓库保留了源作者的 Apache 2.0 声明，并随权重提供 Google Gemma 条款和 NOTICE。
