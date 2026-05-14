# LLM Cognition Benchmark

Systematic evaluation of **human-like cognitive abilities** in large language models across 5 leading Chinese LLMs (2026).

## Dimensions

| Dimension | Dataset(s) | Task | Samples |
|-----------|-----------|------|---------|
| **Commonsense Reasoning** | CommonsenseQA, PIQA, Social IQa, Commonsense-CN | Multi-choice (zero-shot) | 250 × 4 |
| **Emotional Intelligence** | EmoBench (EA + EU) | Scenario-based selection | 200 × 2 |
| **Humor Understanding** | Chumor | Binary explanation judgment | 3,339 |

## Models Evaluated

| Model | Parameters | API Provider |
|-------|-----------|-------------|
| DeepSeek-V3 | 671B MoE | DeepSeek |
| Qwen2.5-72B-Instruct | 72B | SiliconFlow |
| Kimi K2 | MoE | SiliconFlow |
| GLM-4-32B | 32B | SiliconFlow |
| Hunyuan-A13B | 13B | SiliconFlow |

## Quick Results

### Commonsense Reasoning

| Model | CommonsenseQA | PIQA | Social IQa | CN (zh) |
|-------|:-----------:|:----:|:----------:|:------:|
| **Kimi K2** | **85.6%** | 93.2% | **82.8%** | 76.0% |
| Qwen2.5-72B | 84.0% | **95.2%** | 81.2% | **78.0%** |
| GLM-4-32B | 84.0% | 89.2% | 82.0% | 72.7% |
| DeepSeek-V3 | 80.4% | 88.8% | 76.8% | 72.0% |
| Hunyuan-A13B | 76.4% | 84.0% | 75.2% | 69.3% |

### Emotional Intelligence

| Model | Emotional Application | Emotional Understanding |
|-------|:--------------------:|:----------------------:|
| **Kimi K2** | **0.765** | **0.591** |
| Qwen2.5-72B | 0.750 | 0.557 |
| DeepSeek-V3 | 0.750 | 0.471 |
| GLM-4-32B | 0.705 | 0.523 |
| Hunyuan-A13B | 0.685 | 0.530 |

### Key Findings

1. PIQA (physical reasoning) is approaching ceiling — Qwen 95.2%, all models > 84%
2. Social reasoning remains the hardest — best only 82.8%
3. Cross-lingual drop (EN→CN): all models lose 6-10% accuracy
4. Kimi K2 leads in 5 of 7 metrics; small models (Hunyuan 13B) still competitive on specific tasks

## Project Structure

```
LLM-Cognition-Benchmark/
├── commonsense/           # Commonsense reasoning evaluation
│   ├── llms/llm_apis.py   # Unified API layer (OpenAI SDK)
│   ├── eval/              # 4 benchmark scripts
│   ├── datas/             # Preprocessed datasets
│   ├── run.py             # One-click run + visualization
│   └── GUIDE_FROM_SCRATCH.md
├── emotional_intelligence/ # Emotional intelligence evaluation
│   ├── eval_unified.py    # Main evaluation script
│   ├── eval_worker.py     # Per-model worker
│   └── src/               # EmoBench data
├── humor/                 # Chinese humor understanding
│   ├── eval_unified.py    # Evaluation script
│   └── README.md
└── reports/
    └── FINAL_REPORT.md    # Full analysis report
```

## Quick Start

```bash
pip install -r requirements.txt
cp commonsense/.env.example commonsense/.env
# Edit .env with your API keys

# Run commonsense evaluation
cd commonsense
PYTHONPATH=. python run.py

# Run emotional intelligence evaluation
cd ../emotional_intelligence
python eval_unified.py
```

## API Architecture

All 5 models are called via **unified OpenAI-compatible SDK**. Each model is one configuration line:

```python
"kimi-k2": {
    "model": "moonshotai/Kimi-K2-Instruct-0905",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "QWEN_API_KEY",
},
```

## Citation

Original benchmark design adapted from *"Evaluating Human-Like Cognition in Large Language Models"* (CSC5051, CUHK-SZ, 2024).

## License

MIT
