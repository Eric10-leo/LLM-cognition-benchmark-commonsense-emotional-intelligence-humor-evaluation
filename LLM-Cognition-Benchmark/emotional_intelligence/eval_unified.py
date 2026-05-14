import os, sys, json, re, numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from CommonSense directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'CommonSense', '.env'))

API_KEY = os.environ.get("QWEN_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"

data_ea = json.load(open(os.path.join(os.path.dirname(__file__), 'data_EA.json'), 'r', encoding='utf-8'))
data_eu = json.load(open(os.path.join(os.path.dirname(__file__), 'data_EU.json'), 'r', encoding='utf-8'))
print(f"Loaded {len(data_ea)} EA samples, {len(data_eu)} EU samples")

# Model name -> (api_key_env, base_url, model_id) - only SiliconFlow models use QWEN_API_KEY
MODEL_CONFIG = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "qwen2.5": ("QWEN_API_KEY", BASE_URL, "Qwen/Qwen2.5-72B-Instruct-128K"),
    "glm4-32b": ("QWEN_API_KEY", BASE_URL, "THUDM/GLM-4-32B-0414"),
    "hunyuan-a13b": ("QWEN_API_KEY", BASE_URL, "tencent/Hunyuan-A13B-Instruct"),
    "kimi-k2": ("QWEN_API_KEY", BASE_URL, "moonshotai/Kimi-K2-Instruct-0905"),
}


def call_model(model_key, prompt, options):
    key_env, base, model_id = MODEL_CONFIG[model_key]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"Missing {key_env}")
    client = OpenAI(api_key=api_key, base_url=base)

    # Format options as A/B/C/D for easy single-letter answer
    letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
    opt_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    full_prompt = f"{prompt}\n\nOptions:\n{opt_str}\n\nJust return the letter of the best option (e.g. 'A'). Do not explain."

    r = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1,
        max_tokens=10,
    )
    ans = r.choices[0].message.content.strip().upper()

    # Try to extract a letter A-D from the response
    m = re.search(r'\b([A-D])\b', ans)
    if m:
        idx = letters.index(m.group(1))
        return idx
    return None


def evaluate_ea():
    print("\n=== EA Evaluation ===")
    records = []
    for mk in MODEL_CONFIG:
        print(f"\nEvaluating {mk} on EA...")
        correct = {"Problem": {}, "Relationship": {}}
        total = {"Problem": {}, "Relationship": {}}
        errors = 0
        for i, item in enumerate(data_ea):
            problem = item['Problem']
            relation = item['Relationship']
            prompt = f"Scenario: {item['Scenario']['en']}\nWhat action should {item['Subject']['en']} take?"
            options = item['Choices']['en']
            label = item['Label']
            try:
                idx = call_model(mk, prompt, options)
            except Exception as e:
                errors += 1
                idx = None
            is_correct = (idx == label)
            total['Problem'].setdefault(problem, 0)
            correct['Problem'].setdefault(problem, 0)
            total['Relationship'].setdefault(relation, 0)
            correct['Relationship'].setdefault(relation, 0)
            total['Problem'][problem] += 1
            total['Relationship'][relation] += 1
            if is_correct:
                correct['Problem'][problem] += 1
                correct['Relationship'][relation] += 1

        scores = {}
        for dim in ['Problem', 'Relationship']:
            accs = [correct[dim][k] / total[dim][k] for k in total[dim]]
            scores[dim] = round(np.mean(accs), 4) if accs else 0.0
        scores['EA'] = round(np.mean([scores['Problem'], scores['Relationship']]), 4)
        print(f"  {mk}: Problem={scores['Problem']:.3f}, Relationship={scores['Relationship']:.3f}, EA={scores['EA']:.3f} (errors={errors})")
        records.append({'model': mk, **scores})

    os.makedirs('results', exist_ok=True)
    json.dump(records, open('results/ea_results.json', 'w'), indent=2)
    return records


def evaluate_eu():
    print("\n=== EU Evaluation ===")
    records = []
    for mk in MODEL_CONFIG:
        print(f"\nEvaluating {mk} on EU...")
        correct = {}
        total = {}
        errors = 0
        for i, item in enumerate(data_eu):
            cat = item['Category']
            tasks = [
                ("Emotion", f"Scenario: {item['Scenario']['en']}\nWhich emotion does {item['Subject']['en']} feel?",
                 item['Emotion']['Choices']['en'], item['Emotion']['Label']['en']),
                ("Cause", f"Scenario: {item['Scenario']['en']}\nWhat is the cause?",
                 item['Cause']['Choices']['en'], item['Cause']['Label']['en']),
            ]
            for task_name, prompt, options, label in tasks:
                try:
                    idx = call_model(mk, prompt, options)
                except Exception as e:
                    errors += 1
                    idx = None
                is_correct = (idx is not None and options[idx] == label)
                total.setdefault((cat, task_name), 0)
                correct.setdefault((cat, task_name), 0)
                total[(cat, task_name)] += 1
                if is_correct:
                    correct[(cat, task_name)] += 1

        scores = {}
        cats = sorted({k[0] for k in total})
        for cat in cats:
            accs = [correct.get((cat, t), 0) / max(total.get((cat, t), 1), 1) for t in ['Emotion', 'Cause']]
            scores[cat] = round(np.mean(accs), 4) if accs else 0.0
        scores['EU'] = round(np.mean(list(scores.values())), 4) if scores else 0.0
        print(f"  {mk}: EU={scores['EU']:.3f} (errors={errors})")
        records.append({'model': mk, **scores})

    json.dump(records, open('results/eu_results.json', 'w'), indent=2)
    return records


if __name__ == '__main__':
    ea_results = evaluate_ea()
    eu_results = evaluate_eu()
    print("\n=== FINAL RESULTS ===")
    print("\nEA:")
    for r in ea_results:
        print(f"  {r['model']:15s} EA={r['EA']:.4f}  (Problem={r['Problem']:.4f}, Relationship={r['Relationship']:.4f})")
    print("\nEU:")
    for r in eu_results:
        print(f"  {r['model']:15s} EU={r['EU']:.4f}")
