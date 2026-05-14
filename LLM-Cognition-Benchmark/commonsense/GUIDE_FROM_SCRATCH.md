# 从零手搓 LLM BenchMark 评测项目指南

## 项目概述

本项目评测大语言模型在**常识推理（Commonsense Reasoning）**上的表现，涵盖 4 个数据集 × 5 个模型，是一个完整的 LLM 评测流水线。

## 第一步：理解评测目标

你要做的是：用同一套题库去考不同的 LLM，看谁得分高。

### 4 个评测数据集

| 数据集 | 考察能力 | 题型 | 语言 | 数据来源 |
|--------|----------|------|------|----------|
| **CommonsenseQA** | 事实常识推理 | 5选1 | 英文 | [Talmor et al., 2019](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) |
| **PIQA** | 物理常识推理 | 2选1 | 英文 | [Bisk et al., 2020](https://leaderboard.allenai.org/physicaliqa) |
| **Social IQa** | 社交常识推理 | 3选1 | 英文 | [Sap et al., 2019](https://leaderboard.allenai.org/socialiqa) |
| **Commonsense-CN** | 跨语言泛化 | 多选 | 中文 | 上述三个数据集的中文翻译子集 |

### 5 个被测模型（2026 版）

| 模型 | API 提供商 | 购买地址 |
|------|-----------|----------|
| DeepSeek-V3 | DeepSeek | https://platform.deepseek.com |
| Qwen2.5-72B / Qwen3 | 阿里百炼 / SiliconFlow | https://siliconflow.cn |
| GLM-5 | 智谱 AI | https://open.bigmodel.cn |
| Hunyuan-Turbo | 腾讯混元 | https://cloud.tencent.com/product/hunyuan |
| Kimi K2 | Moonshot | https://platform.moonshot.cn |

---

## 第二步：准备数据

### 数据下载

原始数据来源：
- CommonsenseQA: https://www.tau-nlp.sites.tau.ac.il/commonsenseqa (dev.jsonl)
- PIQA: https://github.com/ybisk/ybisk.github.io/raw/master/PIQA/data/dev.jsonl
- Social IQa: https://raw.githubusercontent.com/allenai/social-iqa/master/socialiqa-train-dev.zip

你需要在 `datas/original_datas/` 目录下放置原始文件。

### 数据预处理脚本

原始数据格式不统一（jsonl / json / 自定义标签文件），需要预处理成统一的 JSON 格式：

```
输入:
  dev.jsonl (CommonsenseQA 原始)
  PIQA_dev.jsonl + PIQA_dev-labels.lst (PIQA 原始)
  social_i_qa.py + ... (Social IQa 原始)

输出:
  CommonsenseQA_extracted.json
  PIQA_extracted.json  
  Social_IQA_extracted.json
```

每个预处理脚本的核心逻辑（以 CommonsenseQA 为例）：

```python
# datas/CommonsenseQA_processing.py
import json

# 读取原始 jsonl
data = []
with open("original_datas/dev.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        # 提取 question, choices, answerKey
        question = item["question"]["stem"]
        choices = {
            "label": [c["label"] for c in item["question"]["choices"]],
            "text": [c["text"] for c in item["question"]["choices"]]
        }
        answerKey = item.get("answerKey", "")
        data.append({"question": question, "choices": choices, "answerKey": answerKey})

# 保存为标准格式
with open("CommonsenseQA_extracted.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### 中文翻译子集

Commonsense-CN 是用 DeepSeek-R1 将上述三个数据集的题目翻译成中文得到的：

```python
# translate.py
import json
from llms.llm_apis import get_llm_client

def translate_to_chinese(text):
    client, model = get_llm_client("deepseek")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"将以下英文翻译为中文：\n{text}"}],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# 从三个数据集中各抽取一部分，翻译，合并为 Commonsense-CN_extracted.json
```

---

## 第三步：搭建 API 调用层

这是整个项目最核心的工程部分。2026 年几乎所有国内模型都兼容 OpenAI API 格式，因此可以用 OpenAI SDK 统一调用。

### 文件结构

```
CommonSense/
├── .env              # API key 配置
├── llms/
│   └── llm_apis.py   # 统一 API 调用层
├── eval/             # 评测脚本
│   ├── eval_CommonsenseQA.py
│   ├── eval_PIQA.py
│   ├── eval_Social_IQA.py
│   └── eval_Commonsense-CN.py
├── datas/            # 预处理后的数据集
├── results/          # 评测结果
├── run.py            # 一键运行 + 生成图表
└── images.py         # 额外的可视化
```

### llm_apis.py 核心代码

```python
from openai import OpenAI
import os

# 每个模型一行配置：模型名 + 端点 + 环境变量
MODEL_CONFIGS = {
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen3": {
        "model": "Qwen/Qwen2.5-72B-Instruct-128K",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "QWEN_API_KEY",
    },
    # ... 其他模型
}

def get_llm_client(name):
    """返回 (OpenAI client, model_name)，key 缺失则返回 (None, None) 自动跳过"""
    config = MODEL_CONFIGS[name.lower()]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        print(f"[SKIP] Missing API key for {name}")
        return None, None
    return OpenAI(api_key=api_key, base_url=config["base_url"]), config["model"]
```

**为什么这样设计：**
- 统一用 OpenAI SDK 而不是 requests：自动重试、错误处理、类型安全
- 每个模型只是一个字典条目：新增模型一行配置
- 返回 None 时跳过：可以只有部分 key 先跑起来

### .env 文件

```
DEEPSEEK_API_KEY=sk-xxxxxxxx
QWEN_API_KEY=sk-xxxxxxxx
GLM_API_KEY=xxxxxxxx
HUNYUAN_API_KEY=sk-xxxxxxxx
KIMI_API_KEY=sk-xxxxxxxx
```

---

## 第四步：编写评测脚本

每个评测脚本的结构完全相同，以 CommonsenseQA 为例：

```python
# eval/eval_CommonsenseQA.py
import json
from tqdm import tqdm
from dotenv import load_dotenv
from llms.llm_apis import get_llm_client

load_dotenv()

# 1. 加载数据
with open("datas/CommonsenseQA_extracted.json", "r") as f:
    data = json.load(f)[:250]  # 取前250条

# 2. 构造 prompt（每个数据集格式不同）
def build_prompt(question, choices):
    prompt = f"{question}\n"
    for label, text in zip(choices["label"], choices["text"]):
        prompt += f"{label}. {text}\n"
    prompt += "\nPlease select the most appropriate answer from A, B, C, D or E. Just return the letter only.\nAnswer:"
    return prompt

# 3. 评测单个模型
def evaluate_model(model_name):
    client, model = get_llm_client(model_name)
    if client is None:
        return  # key 缺失，跳过

    predictions = []
    correct = 0

    for item in tqdm(data, desc=f"Evaluating {model_name}"):
        prompt = build_prompt(item["question"], item["choices"])
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,    # 低温度，追求确定性和可复现
                max_tokens=10,      # 答案只有一个字母
            )
            model_output = response.choices[0].message.content.strip()
            predicted = model_output[0].upper()  # 取首字母作为预测答案
            label = item["answerKey"]
            is_correct = predicted == label
            if is_correct:
                correct += 1

            predictions.append({
                "question": item["question"],
                "answerKey": label,
                "model_output": model_output,
                "predicted": predicted,
                "correct": is_correct,
            })
        except Exception as e:
            predictions.append({
                "question": item["question"],
                "answerKey": item.get("answerKey", ""),
                "model_output": f"[ERROR: {e}]",
                "predicted": "?",
                "correct": False,
            })

    # 4. 保存结果
    os.makedirs(f"results/{task_name}", exist_ok=True)
    with open(f"results/{task_name}/{model_name}_cqa.json", "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # 5. 保存准确率
    acc = correct / len([x for x in predictions if x["answerKey"]])
    acc_data = {
        "model": model_name,
        "task": task_name,
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(predictions),
    }
    with open(f"results/{task_name}/acc_{model_name}_cqa.json", "w") as f:
        json.dump(acc_data, f, indent=2)

    print(f"[{model_name}] Accuracy: {acc:.2%} ({correct}/{len(predictions)})")

# 6. 主程序
models = ["deepseek", "qwen3", "glm5", "hunyuan", "kimi"]
for model in models:
    evaluate_model(model)
```

### 关键设计决策

| 决策 | 为什么 |
|------|--------|
| temperature=0.1 | 接近确定性输出，确保结果可复现 |
| max_tokens=10 | 答案只需一个字母，限制输出防止模型啰嗦 |
| 取 output[0].upper() | 简单粗暴，但覆盖 95%+ 的情况 |
| 每条保存完整 JSON | 方便后续做 case study，不仅看准确率 |
| try/except 包裹每道题 | 一道题失败不影响其余 |

---

## 第五步：结果汇总与可视化

run.py 负责收集结果并生成图表：

```python
# run.py 的核心流程
# 1. 依次运行所有 eval 脚本
for script in ["eval_CommonsenseQA.py", "eval_Social_IQA.py", "eval_PIQA.py", "eval_Commonsense-CN.py"]:
    subprocess.run(["python", script])

# 2. 收集所有 acc_*.json 文件
acc_results = []
for task_dir in os.listdir("results"):
    for fn in os.listdir(f"results/{task_dir}"):
        if fn.startswith("acc_"):
            acc_results.append(json.load(open(f"results/{task_dir}/{fn}")))

# 3. 按任务生成柱状图 (matplotlib)
# 4. 生成跨任务对比图
# 5. 输出 Markdown 汇总表
```

---

## 第六步：完整操作流程

### 6.1 环境准备

```bash
# Python 3.10+
pip install openai python-dotenv tqdm matplotlib requests numpy
```

### 6.2 获取数据

```bash
cd CommonSense/datas/original_datas

# 下载 CommonsenseQA
wget https://s3.amazonaws.com/commensenseqa/dev.jsonl

# 下载 PIQA
wget https://github.com/ybisk/ybisk.github.io/raw/master/PIQA/data/dev.jsonl -O PIQA_dev.jsonl
wget https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/PIQA/data/dev-labels.lst -O PIQA_dev-labels.lst

# 下载 Social IQa
wget https://raw.githubusercontent.com/allenai/social-iqa/master/socialiqa-train-dev.zip
unzip socialiqa-train-dev.zip
```

### 6.3 预处理数据

```bash
python datas/CommonsenseQA_processing.py
python datas/PIQA_processing.py
python datas/Social_IQA_processing.py
python translate.py  # 生成中文版（可选）
```

### 6.4 配置 API Key

编辑 `.env` 文件，填入各平台的 API key。

### 6.5 运行评测

```bash
# 单独跑一个数据集
PYTHONPATH=. python eval/eval_CommonsenseQA.py

# 或一键跑全部
PYTHONPATH=. python run.py
```

### 6.6 查看结果

- `results/accuracy_summary.md` — 准确率汇总表
- `results/acc_bar_*.png` — 各任务的柱状图
- `results/acc_bar_cross_tasks.png` — 跨任务对比图

---

## 第七步：常见问题

### Q: 为什么不用 LangChain / LlamaIndex？
A: 这个项目的调用逻辑很简单（一个 prompt → 一个 API call → 取答案），不需要复杂的编排框架。直接 OpenAI SDK 最轻量。

### Q: 为什么每数据集只跑 250 条？
A: 节省 API 费用。全量 CommonsenseQA 有 1200+ 条，跑 5 个模型 × 4 个数据集全量需要几千次 API 调用。250 条足以得到统计上有意义的比较。

### Q: 如果模型不返回单个字母怎么办？
A: 当前代码取 `output[0].upper()`，如果模型输出 "The answer is A."，也能正确取到 A。但如果模型用中文回答 "答案是A"，`output[0]` 是 "答" 就会错。可以加更 robust 的解析逻辑：

```python
import re
match = re.search(r'\b([A-E])\b', model_output)
predicted = match.group(1) if match else model_output[0].upper()
```

### Q: 为什么我用 requests 而不是 openai SDK 也能跑？
A: 如果你不想装 openai 包，可以用 requests + 手写 HTTP：

```python
import requests
r = requests.post(
    f"{base_url}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": model, "messages": [{"role": "user", "content": prompt}]}
)
answer = r.json()["choices"][0]["message"]["content"]
```

但 openai SDK 提供了自动重试、流式输出、更好的错误类型等，建议直接用。
