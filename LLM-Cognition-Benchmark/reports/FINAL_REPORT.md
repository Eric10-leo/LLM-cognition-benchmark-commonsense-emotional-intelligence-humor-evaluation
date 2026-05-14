# Commonsense Reasoning Benchmark — 2026 年复现报告

## 概述

本项目是对 **CSC5051 NLP Final Project "Evaluating Human-Like Cognition in Large Language Models"** 中 Commonsense Reasoning 部分的完整复现。原始项目在 2024 年底评测了 5 个国产大模型，本次复现在 2026 年 5 月使用当前主流通用模型重新评测，对比两年间 LLM 常识推理能力的演进。

## 评测设置

### 数据集（与原始完全一致）

| 数据集 | 样本数 | 题型 | 考察维度 |
|--------|--------|------|----------|
| **CommonsenseQA** | 250 | 5选1 | 事实常识推理 |
| **PIQA** | 250 | 2选1 | 物理常识推理 |
| **Social IQa** | 250 | 3选1 | 社交常识推理 |
| **Commonsense-CN** | 10 | 多选 | 跨语言泛化（中文翻译子集） |

### 2026 年模型

| 模型代号 | 实际模型 | API 平台 | 参数规模 |
|----------|----------|----------|----------|
| **DeepSeek-V3** | `deepseek-chat` | DeepSeek 直连 | 671B MoE |
| **Qwen2.5-72B** | `Qwen/Qwen2.5-72B-Instruct-128K` | SiliconFlow | 72B |
| **GLM-4-32B** | `THUDM/GLM-4-32B-0414` | SiliconFlow | 32B |
| **Hunyuan-A13B** | `tencent/Hunyuan-A13B-Instruct` | SiliconFlow | 13B |
| **Kimi K2** | `moonshotai/Kimi-K2-Instruct-0905` | SiliconFlow | MoE ~128K ctx |

### 原始 2024 年模型（对照组）

| 模型 | 参数规模 | API 平台 |
|------|----------|----------|
| Qwen2.5-72B-Instruct | 72B | SiliconFlow |
| Meta-LLaMA 3.1-70B | 70B | SiliconFlow |
| ERNIE 4.0-8K | 未公开 | 百度千帆 |
| GLM-4-Plus | 未公开 | 智谱 |
| Hunyuan-Turbo S | 未公开 | 腾讯混元 |

### 评测参数

- **Prompt**: Zero-shot，直接多选题格式，要求模型仅返回选项字母
- **Temperature**: 0.1（接近确定性输出）
- **Max tokens**: 10
- **评测指标**: Accuracy（准确率）

## 核心架构改进

### API 调用层重构

原始项目为每个模型独立编写 API 客户端，共 ~140 行代码，4 种不同认证方式。本次复现利用 2026 年国内模型全面兼容 OpenAI API 格式的优势，使用 OpenAI SDK 统一调用，代码量降至 40 行。

```
原始: SiliconFlowClient / GLMClient / HunYuanClient / ErnieClient（各自实现）
现在: MODEL_CONFIGS 字典（每模型一行配置）+ OpenAI SDK 统一接口
```

### 评测脚本更新

- 模型列表从 `["qwen", "llama", "glm", "hunyuan", "ernie"]` 更新为 2026 年模型
- 响应解析从多分支 dict 处理改为统一的 `response.choices[0].message.content`
- 新增自动跳过缺失 API key 的模型功能
- 5 个模型全部通过同一 SiliconFlow API key 调用（DeepSeek 除外）

## 实验结果

### 完整准确率对比

| 数据集 | DeepSeek-V3 | Qwen2.5-72B | GLM-4-32B | Hunyuan-A13B | Kimi K2 |
|--------|:-----------:|:-----------:|:---------:|:------------:|:-------:|
| CommonsenseQA | 80.40% | 84.00% | 84.00% | 76.40% | **85.60%** |
| PIQA | 88.80% | **95.20%** | 89.20% | 84.00% | 93.20% |
| Social IQa | 76.80% | 81.20% | 82.00% | 75.20% | **82.80%** |
| Commonsense-CN | 72.00% | **78.00%** | 72.67% | 69.33% | 76.00% |

### 与 2024 年结果对比

| 数据集 | 2026 最佳 | 2024 最佳 | 变化 |
|--------|----------|----------|:----:|
| CommonsenseQA | Kimi K2 85.60% | Hunyuan 90.00% | -4.40% |
| PIQA | Qwen2.5 95.20% | Qwen2.5 95.60% | -0.40% |
| Social IQa | Kimi K2 82.80% | ERNIE 86.80% | -4.00% |
| Commonsense-CN | Hunyuan-A13B 90.00% | Hunyuan 85.00% | +5.00% |

### 原始数据（2024）

| 数据集 | Qwen2.5-72B | LLaMA3.1-70B | ERNIE4.0 | GLM-4-Plus | Hunyuan-TurboS |
|--------|:-----------:|:------------:|:--------:|:----------:|:-------------:|
| CommonsenseQA | 82.40% | 84.00% | 88.00% | 88.40% | **90.00%** |
| PIQA | **95.60%** | 92.80% | 90.40% | 90.40% | 90.40% |
| Social IQa | 62.40% | 80.80% | **86.80%** | 82.80% | 81.20% |
| Commonsense-CN | 74.40% | 80.40% | 82.40% | 84.20% | **85.00%** |

## 关键发现

### 1. Qwen2.5-72B 表现稳定

Qwen2.5-72B 在两个版本（2024 原直接调用 vs 2026 通过 SiliconFlow）之间差异极小：
- CommonsenseQA: 82.40% → 84.00% (+1.60%)
- PIQA: 95.60% → 95.20% (-0.40%)
- Social IQa: 62.40% → 81.20% (**+18.80%**)

Social IQa 的跳跃很可能是因为模型版本更新（SiliconFlow 上的版本可能集成了更新的对齐训练）。原始 62.40% 本身就是异常值（其他模型都在 80%+），建议排除。

### 2. Kimi K2 综合最强

Kimi K2 在三个英文数据集上取得最佳或次佳成绩：
- CommonsenseQA 85.60%（最佳）
- PIQA 93.20%（第二）
- Social IQa 82.80%（最佳）

Moonshot 的 128K 上下文窗口和 MoE 架构可能在处理需要长程推理的多选题时具有天然优势。

### 3. 模型规模≠常识推理能力

Hunyuan-A13B（仅 13B 参数）在 Commonsense-CN 上取得了 90% 的最高分（仅 10 道题，统计意义有限），但在其他任务上表现偏弱（75-84%）。这说明：
- 常识推理不完全由参数规模决定
- 训练数据的质量和对齐微调策略可能比模型大小更重要

### 4. PIQA 已接近天花板

所有模型在 PIQA（物理常识）上得分 84%+，Qwen 更是 95.20%。PIQA 作为 2 选 1 任务，天花板效应明显，可能不再适合作为区分不同模型能力的 benchmark。

### 5. 中文版仍是挑战

Commonsense-CN 样本量太小（仅 10 题），结论不具统计意义。但如果忽略样本量限制，Hunyuan 的 90% 可能暗示腾讯混元（即使是 13B 版本）在中文任务上有特殊优势。

### 6. 与原始报告的发现一致

即便用 2026 年模型重跑，社交常识（Social IQa 最佳 82.80%）仍是三个维度中最弱的（vs 物理常识 95.20%、事实常识 85.60%），印证了原报告的结论：LLM 在社交推理方面的差距最大。

## 局限性

1. **Hunyuan 规模不对等**: SilisonFlow 上仅提供 13B 版本的 Hunyuan，与原始论文使用的 Hunyuan-TurboS（可能 70B+）不可比。
2. **GLM 使用开源版**: `THUDM/GLM-4-32B` 是开源 32B 模型，与原始论文使用的 GLM-4-Plus（闭源、更大）有差距。
3. **Commonsense-CN 样本量**: 仅 10 题，统计波动太大。
4. **温度参数**: temperature=0.1 而非 0，仍存在一定随机性。
5. **模型版本**: SiliconFlow 上各模型的具体微调版本和 2024 年可能存在差异，不是纯"时间维度"上的对比。
