import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from llms.llm_apis import get_llm_client

load_dotenv()

with open("datas/Social_IQA_extracted.json", "r", encoding="utf-8") as f:
    data = json.load(f)[:250]


def build_prompt(question, choices):
    labels = choices["label"]
    texts = choices["text"]
    prompt = f"{question}\n"
    for label, choice in zip(labels, texts):
        prompt += f"{label}. {choice}\n"
    prompt += "\nPlease select the most appropriate answer from A, B, or C. Just return the letter only.\nAnswer:"
    return prompt


def evaluate_model(model_name):
    client, model = get_llm_client(model_name)
    if client is None:
        return
    predictions = []
    correct = 0

    for item in tqdm(data, desc=f"Evaluating {model_name}"):
        prompt = build_prompt(item["question"], item["choices"])
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

    os.makedirs("results/Social_IQA", exist_ok=True)
    with open(f"results/Social_IQA/{model_name}_socialiqa.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    acc = correct / len([x for x in predictions if x["answerKey"]])
    acc_data = {
        "model": model_name,
        "task": "Social_IQA",
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(predictions),
    }
    with open(f"results/Social_IQA/acc_{model_name}_socialiqa.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2)

    print(f"[{model_name}] Social_IQA Accuracy: {acc:.2%} ({correct}/{len(predictions)})")


models = ["deepseek", "qwen3", "glm5", "hunyuan", "kimi"]

if __name__ == "__main__":
    for model in models:
        evaluate_model(model)
