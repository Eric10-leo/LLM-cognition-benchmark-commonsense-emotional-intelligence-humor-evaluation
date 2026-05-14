from openai import OpenAI
import os

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
    "glm5": {
        "model": "THUDM/GLM-4-32B-0414",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "QWEN_API_KEY",
    },
    "hunyuan": {
        "model": "tencent/Hunyuan-A13B-Instruct",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "QWEN_API_KEY",
    },
    "kimi": {
        "model": "moonshotai/Kimi-K2-Instruct-0905",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "QWEN_API_KEY",
    },
}


def get_llm_client(name):
    name = name.lower()
    if name not in MODEL_CONFIGS:
        print(f"[SKIP] Unknown model: {name}")
        return None, None

    config = MODEL_CONFIGS[name]
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        print(f"[SKIP] Missing API key for {name} ({config['api_key_env']})")
        return None, None

    return OpenAI(api_key=api_key, base_url=config["base_url"]), config["model"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    for name in MODEL_CONFIGS:
        print(f"\n{'='*40}\nTesting {name.upper()}")
        try:
            client, model = get_llm_client(name)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'hello' in one word."}],
                max_tokens=10,
                temperature=0,
            )
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error: {e}")
