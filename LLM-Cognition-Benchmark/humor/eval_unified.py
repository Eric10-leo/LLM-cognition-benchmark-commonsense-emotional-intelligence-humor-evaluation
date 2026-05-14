import os, sys, time, pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from CommonSense
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'CommonSense', '.env'))

API_KEY = os.environ.get("QWEN_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"

MODEL_CONFIG = {
    "deepseek":    ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "qwen2.5":     ("QWEN_API_KEY", BASE_URL, "Qwen/Qwen2.5-72B-Instruct-128K"),
    "glm4-32b":    ("QWEN_API_KEY", BASE_URL, "THUDM/GLM-4-32B-0414"),
    "hunyuan-a13b":("QWEN_API_KEY", BASE_URL, "tencent/Hunyuan-A13B-Instruct"),
    "kimi-k2":     ("QWEN_API_KEY", BASE_URL, "moonshotai/Kimi-K2-Instruct-0905"),
}

PROMPT_TEMPLATE = """你将看到一个笑话以及对这个笑话的解释。
请判断这个解释是否完全解释了笑话。根据判断，选择"完全解释"或"部分/没有解释"，不需要解释为什么对或者不对。
完全解释输出"good"，部分/没有解释输出"bad"。

笑话：{joke}
笑话解释：{explanation}"""


def evaluate_model(model_key, input_tsv, output_tsv):
    key_env, base_url, model_id = MODEL_CONFIG[model_key]
    api_key = os.environ.get(key_env)
    if not api_key:
        print(f"[SKIP] {model_key}: missing API key ({key_env})")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"\nEvaluating {model_key} ({model_id})...")
    df = pd.read_csv(input_tsv, sep='\t')
    df["Label_2"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=model_key):
        prompt = PROMPT_TEMPLATE.format(joke=row["Joke"], explanation=row["Explanation"])
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            result = r.choices[0].message.content.strip().lower()
            if "good" in result:
                label = "good"
            elif "bad" in result:
                label = "bad"
            else:
                label = "bad"  # default
            df.at[idx, "Label_2"] = label
        except Exception as e:
            print(f"  Error row {idx}: {e}")
            df.at[idx, "Label_2"] = "error"
        time.sleep(0.3)

    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Saved: {output_tsv}")

    # Calculate accuracy
    valid = df[df["Label_2"] != "error"]
    acc = (valid["Label"] == valid["Label_2"]).sum() / len(valid) if len(valid) > 0 else 0
    print(f"  {model_key} Accuracy: {acc:.4f} ({int(acc*len(valid))}/{len(valid)})")
    return acc


if __name__ == "__main__":
    INPUT = "chumor.2.0.tsv"  # Obtain from https://dnaihao.github.io/Chumor-dataset/

    if not os.path.exists(INPUT):
        print(f"ERROR: {INPUT} not found!")
        print("The Chumor dataset is not publicly available in this repo.")
        print("Contact the authors at https://dnaihao.github.io/Chumor-dataset/ to obtain it.")
        sys.exit(1)

    results = {}
    for mk in MODEL_CONFIG:
        acc = evaluate_model(mk, INPUT, f"chumor_{mk}_labeled.tsv")
        if acc is not None:
            results[mk] = acc

    print("\n=== HUMOR RESULTS ===")
    for mk, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {mk:15s} {acc:.4f}")
