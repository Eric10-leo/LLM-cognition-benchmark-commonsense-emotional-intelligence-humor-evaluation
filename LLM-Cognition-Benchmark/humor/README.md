# Chinese Humor Understanding (Chumor)

## Status

The unified evaluation script (`eval_unified.py`) is ready. The **Chumor dataset** (`chumor.2.0.tsv`) is not included in this repo — it must be obtained from the original authors.

## Dataset

Chumor: a Chinese humor understanding dataset from Ruo Zhi Ba (弱智吧).

- **Source**: https://dnaihao.github.io/Chumor-dataset/
- **Format**: TSV with columns `Joke`, `Explanation`, `Label` (good/bad)
- **Size**: 3,339 instances

## Usage

```bash
# 1. Place chumor.2.0.tsv in this directory
# 2. Run evaluation
python eval_unified.py
```

## Original Results (2024)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| GLM-4-Plus | 54.5% | 0.488 | 0.953 | 0.645 |
| Hunyuan | 50.3% | 0.465 | 0.946 | 0.623 |
| Qwen2.5-72B | 49.4% | 0.461 | 0.972 | 0.626 |
| ERNIE 4.0 | 47.3% | 0.452 | 0.993 | 0.621 |
| LLaMA 3.1 | 46.0% | 0.446 | 0.997 | 0.616 |

All models exhibit strong "good" bias — they tend to accept explanations as valid rather than critically judge them.
