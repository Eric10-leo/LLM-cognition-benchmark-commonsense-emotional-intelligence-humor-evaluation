[README.md](https://github.com/user-attachments/files/27739798/README.md)
# Human-like-LLM-benchmark

<p align="center">
  <b>Can machines think like humans?</b><br>
  A multi-dimensional evaluation of 5 leading Chinese LLMs on<br>
  <b>Commonsense Reasoning</b> · <b>Emotional Intelligence</b> · <b>Humor Understanding</b>
</p>

---

## Why This Matters

LLMs can pass bar exams, write code, and translate languages. But can they **understand a joke**? Can they **sense when someone is hurt**? Can they **reason about everyday situations** that any human child handles effortlessly?

These *human-like cognitive abilities* are what separate pattern-matching from genuine intelligence. This project systematically probes three dimensions:

| Dimension | What we test | Why it's hard |
|-----------|-------------|---------------|
| **Commonsense Reasoning** | "Where do you store socks?" | Requires world knowledge no textbook teaches |
| **Emotional Intelligence** | "Sarah's brother is being bullied..." | Requires perspective-taking and social nuance |
| **Humor Understanding** | Chinese jokes from 弱智吧 | Requires cultural context, wordplay, irony detection |

---

## Key Results (2026)

### Commonsense Reasoning

Evaluated on 4 benchmarks with zero-shot prompting across 5 models:

| Model | CommonsenseQA<br><sub>factual</sub> | PIQA<br><sub>physical</sub> | Social IQa<br><sub>social</sub> | Commonsense-CN<br><sub>Chinese</sub> |
|-------|:---:|:---:|:---:|:---:|
| **Kimi K2** | **85.6%** | 93.2% | **82.8%** | 76.0% |
| **Qwen2.5-72B** | 84.0% | **95.2%** | 81.2% | **78.0%** |
| GLM-4-32B | 84.0% | 89.2% | 82.0% | 72.7% |
| DeepSeek-V3 | 80.4% | 88.8% | 76.8% | 72.0% |
| Hunyuan-A13B | 76.4% | 84.0% | 75.2% | 69.3% |
| *2024 best* | *90.0%* | *95.6%* | *86.8%* | *85.0%* |

**Finding:** Social reasoning is the Achilles' heel — even the best model reaches only 82.8%. Physical reasoning (PIQA) is nearing ceiling at 95%.

### Emotional Intelligence

Evaluated on EmoBench — 400 scenarios across Emotional Understanding (EU) and Emotional Application (EA):

| Model | EA<br><sub>knowing what to do</sub> | EU<br><sub>knowing what they feel</sub> | Best Category |
|-------|:---:|:---:|------|
| **Kimi K2** | **0.765** | **0.591** | vocal_cues 0.808 |
| Qwen2.5-72B | 0.750 | 0.557 | emotion_transition 0.733 |
| DeepSeek-V3 | 0.750 | 0.471 | emotion_transition 0.633 |
| GLM-4-32B | 0.705 | 0.523 | vocal_cues 0.808 |
| Hunyuan-A13B | 0.685 | 0.530 | vocal_cues 0.808 |
| *2024 best* | *0.740* | *0.529* | — |

**Finding:** For the first time, EA scores exceed 2024 benchmarks (Kimi +2.5%). But EU remains stubbornly difficult — recognizing *why* someone feels something is still much harder than knowing *what* to do about it.

### Humor Understanding

Chinese humor understanding via Chumor dataset (3,339 jokes from 弱智吧):

| Model | Accuracy | Precision | Recall | F1 |
|-------|:---:|:---:|:---:|:---:|
| *GLM-4-Plus* | 54.5% | 0.488 | 0.953 | 0.645 |
| *Hunyuan-TurboS* | 50.3% | 0.465 | 0.946 | 0.623 |
| *Qwen2.5-72B* | 49.4% | 0.461 | 0.972 | 0.626 |
| *ERNIE 4.0* | 47.3% | 0.452 | 0.993 | 0.621 |
| *LLaMA 3.1-70B* | 46.0% | 0.446 | 0.997 | 0.616 |

*Results from original 2024 evaluation. 2026 re-evaluation pending dataset acquisition.*

**Finding:** Barely above random guessing (50%). All models suffer from extreme "say yes" bias — they accept explanations rather than critically judge them.

---

## Architecture

### The Problem (2024)
Each model required a **separate API client** — 4 authentication methods, 4 request formats, ~140 lines of code:

```python
class SiliconFlowClient: ...  # Bearer token
class GLMClient: ...          # JWT via ZhipuAI SDK
class HunYuanClient: ...      # Tencent auth
class ErnieClient: ...        # AK/SK → access_token → Bearer
```

### The Solution (2026)
All leading Chinese LLMs now support OpenAI-compatible APIs. **One SDK to rule them all** — 40 lines:

```python
MODEL_CONFIGS = {
    "deepseek":    {"model": "deepseek-chat",                     "base_url": "https://api.deepseek.com/v1"},
    "qwen2.5":     {"model": "Qwen/Qwen2.5-72B-Instruct-128K",   "base_url": "https://api.siliconflow.cn/v1"},
    "glm4-32b":    {"model": "THUDM/GLM-4-32B-0414",             "base_url": "https://api.siliconflow.cn/v1"},
    "hunyuan-a13b":{"model": "tencent/Hunyuan-A13B-Instruct",    "base_url": "https://api.siliconflow.cn/v1"},
    "kimi-k2":     {"model": "moonshotai/Kimi-K2-Instruct-0905", "base_url": "https://api.siliconflow.cn/v1"},
}
# That's it. OpenAI(api_key=..., base_url=...) handles the rest.
```

Adding a new model is **one line**. Swapping providers is **one line**. The entire evaluation pipeline is provider-agnostic.

---

## Project Structure

```
LLM-Cognition-Benchmark/
├── README.md
├── requirements.txt
├── commonsense/                     # ✅ Complete + Re-evaluated (2026)
│   ├── llms/llm_apis.py             # Unified API layer
│   ├── eval/                        # 4 benchmark scripts
│   ├── datas/                       # Preprocessed datasets
│   ├── run.py                       # One-click evaluation + charts
│   └── GUIDE_FROM_SCRATCH.md        # "Build from zero" tutorial
├── emotional_intelligence/          # ✅ Complete + Re-evaluated (2026)
│   ├── eval_unified.py              # Sequential runner
│   ├── eval_worker.py               # Parallel per-model worker
│   └── src/                         # EmoBench dataset
├── humor/                           # ⏳ Script ready, dataset pending
│   ├── eval_unified.py
│   └── README.md
└── reports/
    └── FINAL_REPORT.md              # Full analysis with 2024 vs 2026 comparison
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure API keys
cp commonsense/.env.example commonsense/.env
# Edit .env: add DEEPSEEK_API_KEY and QWEN_API_KEY (SiliconFlow)

# 3. Run commonsense evaluation
cd commonsense
PYTHONPATH=. python run.py
# → Generates results/accuracy_summary.md + 5 PNG charts

# 4. Run emotional intelligence
cd ../emotional_intelligence
python eval_unified.py
```

---

## What We Found

1. **Social intelligence is the final frontier.** Across all three dimensions, models struggle most with tasks requiring social reasoning (Social IQa: 82.8% vs PIQA: 95.2%).

2. **"Yes-man" bias in humor.** Models accept explanations as valid 95%+ of the time — they cannot think critically about whether a joke explanation actually makes sense.

3. **Cross-lingual fragility is real.** Even the strongest models drop 6-10% when tested on the Chinese-translated Commonsense-CN. Surface-level pattern matching ≠ genuine reasoning.

4. **Architecture matters more than scale.** Hunyuan-A13B (13B params) outperforms DeepSeek-V3 (671B MoE) on emotional understanding (EU: 0.530 vs 0.471), proving that training data quality and alignment outweigh raw parameter count.

5. **PIQA is solved.** With all models scoring 84%+ and Qwen at 95.2%, 2-choice physical reasoning no longer discriminates between models. The field needs harder benchmarks.

---

## Models

| Model | Parameters | Provider | Evaluated |
|-------|:---:|---|:---:|
| DeepSeek-V3 | 671B MoE | DeepSeek API | ✅ |
| Qwen2.5-72B-Instruct | 72B | SiliconFlow | ✅ |
| Kimi K2 | MoE | SiliconFlow | ✅ |
| GLM-4-32B | 32B | SiliconFlow | ✅ |
| Hunyuan-A13B | 13B | SiliconFlow | ✅ |

---

*Original benchmark design: CSC5051 NLP Final Project, CUHK-SZ (2024)*  
*2026 re-evaluation & unified framework: built from scratch on SiliconFlow + DeepSeek APIs*
