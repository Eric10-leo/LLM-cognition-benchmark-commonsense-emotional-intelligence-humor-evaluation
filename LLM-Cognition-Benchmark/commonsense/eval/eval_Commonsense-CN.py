import os
import json
import random
from tqdm import tqdm
from dotenv import load_dotenv
from llms.llm_apis import get_llm_client

load_dotenv()

DATA_PATH = "datas/Commonsense-CN_extracted.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    data = random.sample(data, min(250, len(data)))


def build_zh_prompt(question, choices):
    labels = choices["label"]
    texts = choices["text"]
    prompt = f"{question}\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"
    prompt += "\n请选择最合适的选项，仅返回选项字母即可（例如 A、B、C...）。\n答案："
    return prompt


def evaluate_model(model_name):
    client, model = get_llm_client(model_name)
    if client is None:
        return
    predictions = []
    correct = 0

    for item in tqdm(data, desc=f"Evaluating {model_name} on CN"):
        prompt = build_zh_prompt(item["question"], item["choices"])
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            model_output = response.choices[0].message.content.strip()
            predicted = model_output[0].upper()
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

    os.makedirs("results/Commonsense-CN", exist_ok=True)
    with open(f"results/Commonsense-CN/{model_name}_cnqa.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    acc = correct / len([x for x in predictions if x["answerKey"]])
    acc_data = {
        "model": model_name,
        "task": "Commonsense-CN",
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(predictions),
    }
    with open(f"results/Commonsense-CN/acc_{model_name}_cnqa.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2)

    print(f"[{model_name}] Commonsense-CN Accuracy: {acc:.2%} ({correct}/{len(predictions)})")


models = ["deepseek", "qwen3", "glm5", "hunyuan", "kimi"]

if __name__ == "__main__":
    for model in models:
        evaluate_model(model)
