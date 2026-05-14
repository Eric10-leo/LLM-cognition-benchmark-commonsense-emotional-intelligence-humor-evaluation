import os, sys, json, re, numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'CommonSense', '.env'))
API_KEY = os.environ.get("QWEN_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"

data_ea = json.load(open('data_EA.json', 'r', encoding='utf-8'))
data_eu = json.load(open('data_EU.json', 'r', encoding='utf-8'))

MODEL_CONFIG = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "qwen2.5": ("QWEN_API_KEY", BASE_URL, "Qwen/Qwen2.5-72B-Instruct-128K"),
    "glm4-32b": ("QWEN_API_KEY", BASE_URL, "THUDM/GLM-4-32B-0414"),
    "hunyuan-a13b": ("QWEN_API_KEY", BASE_URL, "tencent/Hunyuan-A13B-Instruct"),
    "kimi-k2": ("QWEN_API_KEY", BASE_URL, "moonshotai/Kimi-K2-Instruct-0905"),
}


def call_model(mk, prompt, options):
    key_env, base, model_id = MODEL_CONFIG[mk]
    client = OpenAI(api_key=os.environ.get(key_env), base_url=base)
    letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
    opt_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    full = f"{prompt}\n\nOptions:\n{opt_str}\n\nJust return the letter of the best option (e.g. 'A'). Do not explain."
    r = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": full}], temperature=0.1, max_tokens=10)
    ans = r.choices[0].message.content.strip().upper()
    m = re.search(r'\b([A-D])\b', ans)
    return letters.index(m.group(1)) if m else None


def run_ea(mk):
    correct = {"Problem": {}, "Relationship": {}}
    total = {"Problem": {}, "Relationship": {}}
    for item in data_ea:
        p, r, lab = item['Problem'], item['Relationship'], item['Label']
        prompt = f"Scenario: {item['Scenario']['en']}\nWhat action should {item['Subject']['en']} take?"
        try:
            idx = call_model(mk, prompt, item['Choices']['en'])
        except:
            idx = None
        total['Problem'].setdefault(p, 0); correct['Problem'].setdefault(p, 0)
        total['Relationship'].setdefault(r, 0); correct['Relationship'].setdefault(r, 0)
        total['Problem'][p] += 1; total['Relationship'][r] += 1
        if idx == lab: correct['Problem'][p] += 1; correct['Relationship'][r] += 1
    scores = {}
    for dim in ['Problem', 'Relationship']:
        scores[dim] = round(np.mean([correct[dim][k] / total[dim][k] for k in total[dim]]), 4)
    scores['EA'] = round(np.mean([scores['Problem'], scores['Relationship']]), 4)
    return scores


def run_eu(mk):
    correct, total = {}, {}
    for item in data_eu:
        cat = item['Category']
        for t, prompt, opts, lab in [
            ("Emotion", f"Scenario: {item['Scenario']['en']}\nWhich emotion does {item['Subject']['en']} feel?", item['Emotion']['Choices']['en'], item['Emotion']['Label']['en']),
            ("Cause", f"Scenario: {item['Scenario']['en']}\nWhat is the cause?", item['Cause']['Choices']['en'], item['Cause']['Label']['en'])]:
            try:
                idx = call_model(mk, prompt, opts)
            except:
                idx = None
            total.setdefault((cat, t), 0); correct.setdefault((cat, t), 0)
            total[(cat, t)] += 1
            if idx is not None and opts[idx] == lab: correct[(cat, t)] += 1
    scores = {}
    for cat in sorted({k[0] for k in total}):
        scores[cat] = round(np.mean([correct.get((cat, t), 0) / max(total.get((cat, t), 1), 1) for t in ['Emotion', 'Cause']]), 4)
    scores['EU'] = round(np.mean(list(scores.values())), 4) if scores else 0.0
    return scores


if __name__ == '__main__':
    mk = sys.argv[1]
    task = sys.argv[2]
    if task == 'ea':
        s = run_ea(mk)
        fn = f'results/ea_{mk}.json'
    else:
        s = run_eu(mk)
        fn = f'results/eu_{mk}.json'
    os.makedirs('results', exist_ok=True)
    json.dump(s, open(fn, 'w'), indent=2)
    print(f'{mk} {task.upper()}: {json.dumps(s)}')
