import os
from typing import Any, Dict

from llm_providers import create_llm


def run_case(name: str, cfg: Dict[str, Any], prompt: str) -> None:
    """Run one backend; skip/print errors gracefully."""
    print(f"\n=== {name} ===")
    try:
        llm = create_llm(cfg)
        out = llm.generate(prompt)
        print(out.text)
    except Exception as exc:  # pragma: no cover - manual script
        print(f"[{name}] failed: {exc}")


def transformers_case():
    """本地 transformers（需已装 torch/transformers 与模型）。"""
    model = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/30b8421510892303dc5ddd6cd0ac90ca2053478d"
    cfg = {
        "type": "transformers",
        "llm_path": model,
        "llm_device": "cuda",
        "max_new_tokens": 32,
    }
    run_case("transformers", cfg, "用一句话介绍 Transformer 模型。")


def openai_chat_case():
    """OpenAI 兼容接口（官方或兼容服务）。"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    if not api_key:
        print("skip openai_chat_case: set OPENAI_API_KEY env to run")
        return
    cfg = {
        "type": "openai",
        "llm_name": model,
        "api_key": api_key,
        "api_base": api_base,
        "max_new_tokens": 64,
    }
    run_case("openai_chat", cfg, "给出 3 条高效学习技巧。")


def requests_case():
    """requests 直连 HTTP（如 vLLM OpenAI 兼容 server）。"""
    api_url = os.getenv("REQUESTS_API_URL") or "http://localhost:8001/v1/chat/completions"
    api_key = os.getenv("REQUESTS_API_KEY")
    model = os.getenv("REQUESTS_MODEL")
    cfg = {
        "type": "requests",
        "api_url": api_url,
        "api_headers": {"Authorization": f"Bearer {api_key}"} if api_key else {},
        "chat_mode": "chat",  # 连接 /chat/completions 时开启
        "model": model,
        "max_new_tokens": 64,
    }
    run_case("requests_chat", cfg, "说一个关于 AI 的冷知识。")


def vllm_offline_case():
    """本地 vLLM 离线推理（需安装 vllm 且模型可用）。"""
    model = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/30b8421510892303dc5ddd6cd0ac90ca2053478d"
    if not model:
        print("skip vllm_offline_case: set VLLM_MODEL_PATH env to run")
        return
    cfg = {
        "type": "vllm_offline",
        "llm_path": model,
        "max_new_tokens": 32,
    }
    run_case("vllm_offline", cfg, "简述大语言模型推理的瓶颈。")


if __name__ == "__main__":
    # transformers_case()
    # openai_chat_case()
    requests_case()
    vllm_offline_case()