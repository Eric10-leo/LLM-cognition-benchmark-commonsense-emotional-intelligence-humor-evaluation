import json, random, os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

input_files = [
    "datas/CommonsenseQA_extracted.json",
    "datas/PIQA_extracted.json",
    "datas/Social_IQA_extracted.json"
]
sample_size = 50  # per file = 150 total

def translate(text):
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"请将以下英文翻译为流畅自然的中文，只返回译文，不要额外解释：\n\n{text.strip()}"}],
        temperature=0.3, max_tokens=256,
    )
    return r.choices[0].message.content.strip()

output_data = []
for path in input_files:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = random.sample(data, min(len(data), sample_size))
    print(f"{path}: translating {len(samples)} items...")
    for item in samples:
        new_item = item.copy()
        try:
            new_item['question'] = translate(item['question'])
            if 'choices' in item and 'text' in item['choices']:
                new_item['choices']['text'] = [translate(c) for c in item['choices']['text']]
            output_data.append(new_item)
        except Exception as e:
            print(f"  skip: {e}")
            continue
        time.sleep(0.3)  # rate limit

output_path = "datas/Commonsense-CN_extracted.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print(f"Done: {len(output_data)} items -> {output_path}")
