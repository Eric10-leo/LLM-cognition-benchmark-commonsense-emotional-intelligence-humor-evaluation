# Emotional Intelligence Evaluation (EmoBench)

## Tasks

- **Emotional Application (EA)**: Given a social scenario, select the appropriate action from 4 options (200 samples)
- **Emotional Understanding (EU)**: Identify the emotion a person feels and its cause (200 samples × 2 questions, 11 categories)

## Run

```bash
# All 5 models in sequence
python eval_unified.py

# OR parallel workers (faster)
for m in deepseek qwen2.5 glm4-32b hunyuan-a13b kimi-k2; do
    PYTHONPATH=../commonsense python eval_worker.py $m ea > logs/${m}_ea.log 2>&1 &
    PYTHONPATH=../commonsense python eval_worker.py $m eu > logs/${m}_eu.log 2>&1 &
done
```

## Results (2026)

### EA Scores

| Model | Problem | Relationship | **EA Overall** |
|-------|:-------:|:------------:|:-------------:|
| Kimi K2 | 0.765 | 0.765 | **0.765** |
| DeepSeek-V3 | 0.750 | 0.750 | 0.750 |
| Qwen2.5-72B | 0.750 | 0.750 | 0.750 |
| GLM-4-32B | 0.705 | 0.705 | 0.705 |
| Hunyuan-A13B | 0.685 | 0.685 | 0.685 |

### EU Scores

| Model | **EU Overall** |
|-------|:-------------:|
| Kimi K2 | **0.591** |
| Qwen2.5-72B | 0.557 |
| Hunyuan-A13B | 0.530 |
| GLM-4-32B | 0.523 |
| DeepSeek-V3 | 0.471 |

## Data

EmoBench dataset from *"EmoBench: Evaluating Emotional Intelligence of Large Language Models"*. Covers 11 emotion categories including emotion_transition, false_belief, faux_pas, perspective_taking, etc.
