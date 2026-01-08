import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as torch_mod


@dataclass
class GenerateResult:
    """Standard generation return payload."""

    text: str
    raw: Any = None


def _import_module(module: str, package: str) -> Any:
    """Import a module lazily with a clear error if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise ImportError(
            f"{package} is required for this LLM backend. Please install `{package}`."
        ) from exc


def _get_torch():
    return _import_module("torch", "torch")


def _get_transformers():
    transformers = _import_module("transformers", "transformers")
    return (
        transformers.AutoModelForCausalLM,
        transformers.AutoTokenizer,
        transformers.pipeline,
    )


def _get_openai_client():
    openai_mod = _import_module("openai", "openai")
    return openai_mod.OpenAI


def _get_vllm():
    vllm_mod = _import_module("vllm", "vllm")
    return vllm_mod.LLM, vllm_mod.SamplingParams


def _get_requests():
    return _import_module("requests", "requests")


def _to_torch_dtype(dtype: Optional[str]) -> "torch_mod.dtype":
    """Map config string to torch dtype."""
    torch = _get_torch()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get((dtype or "").lower(), torch.float16)


def _normalize_device(device: Optional[str]) -> "torch_mod.device":
    """Normalize device string into torch.device."""
    torch = _get_torch()
    if device is None:
        return torch.device("cpu")
    if isinstance(device, str):
        if device.startswith("cuda") or device.startswith("gpu"):
            if ":" in device:
                return torch.device(device)
            return torch.device("cuda:0")
        if device == "cpu":
            return torch.device("cpu")
    try:
        return torch.device(device)
    except Exception:
        logger.warning("Unknown device %s, fallback to cpu.", device)
        return torch.device("cpu")


class BaseLLM(ABC):
    """Abstract base class for inference backends."""

    def __init__(
        self, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> GenerateResult:
        """Generate text based on prompt."""

    def predict_options(
        self, prompt: str, options: List[str], **kwargs
    ) -> Dict[str, float]:
        """Return probability-like scores for candidate options."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support logits.")

    def cleanup(self) -> None:
        """Release resources if needed."""
        return None


class TransformersLLM(BaseLLM):
    """HuggingFace transformers backend."""

    def __init__(self, model_name: str, device: Optional[str], **kwargs) -> None:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        super().__init__(max_new_tokens, temperature, top_p)

        self.device = _normalize_device(device)
        self.torch_dtype = _to_torch_dtype(kwargs.get("torch_dtype"))
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.model_kwargs = kwargs.get("model_kwargs") or {}
        self.tokenizer_kwargs = kwargs.get("tokenizer_kwargs") or {}

        torch = _get_torch()
        AutoModelForCausalLM, AutoTokenizer, pipe = _get_transformers()

        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get("tokenizer_name", model_name),
            trust_remote_code=self.trust_remote_code,
            **self.tokenizer_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        # pipeline device expects an int index or -1 for cpu
        device_index = -1 if self.device.type == "cpu" else self.device.index or 0
        self.generator = pipe(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_index,
            torch_dtype=self.torch_dtype,
        )

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> GenerateResult:
        tokens = max_new_tokens or self.max_new_tokens
        outputs = self.generator(
            prompt,
            max_new_tokens=tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs,
        )
        generated = outputs[0]["generated_text"]
        text = generated[len(prompt) :] if generated.startswith(prompt) else generated
        return GenerateResult(text=text, raw=outputs)

    def predict_options(
        self, prompt: str, options: List[str], **kwargs
    ) -> Dict[str, float]:
        torch = _get_torch()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        scores = []
        for opt in options:
            token_id = self.tokenizer(opt).input_ids[-1]
            scores.append(logits[token_id].float())
        scores_tensor = torch.stack(scores)
        probs = torch.nn.functional.softmax(scores_tensor, dim=0).cpu().numpy().tolist()
        return {opt: prob for opt, prob in zip(options, probs)}

    def cleanup(self) -> None:
        try:
            torch = _get_torch()
            del self.generator
            del self.model
            del self.tokenizer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Cleanup failed: %s", exc)


class OpenAIChatLLM(BaseLLM):
    """OpenAI compatible chat backend (OpenAI or vLLM OpenAI API)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> None:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        super().__init__(max_new_tokens, temperature, top_p)
        OpenAI = _get_openai_client()
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> GenerateResult:
        tokens = max_new_tokens or self.max_new_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = response.choices[0].message.content or ""
        return GenerateResult(text=content, raw=response)

    def predict_options(
        self, prompt: str, options: List[str], **kwargs
    ) -> Dict[str, float]:
        # Use completion endpoint to request logprobs for the next token.
        torch = _get_torch()
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=max(len(options), 5),
        )
        top_logprobs = response.choices[0].logprobs.top_logprobs[0]
        token_logprob = {}
        for token_info in top_logprobs:
            token_logprob[token_info.token.strip()] = token_info.logprob

        scores = []
        for opt in options:
            scores.append(token_logprob.get(opt, float("-inf")))
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.nn.functional.softmax(scores_tensor, dim=0).cpu().numpy().tolist()
        return {opt: prob for opt, prob in zip(options, probs)}


class RequestsLLM(BaseLLM):
    """Simplest HTTP backend via requests (assumes a compatible vLLM API)."""

    def __init__(
        self,
        api_url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        super().__init__(max_new_tokens, temperature, top_p)
        self.api_url = api_url
        self.headers = headers or {}
        self.model = kwargs.get("model")
        # auto-detect chat mode if calling OpenAI-like chat endpoint
        self.chat_mode = kwargs.get("chat_mode") or (
            "chat" if "chat/completions" in api_url else "completion"
        )
        # Allow passing default request kwargs (e.g., timeout) from config
        self.request_kwargs = kwargs.get("request_kwargs", {})

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> GenerateResult:
        requests = _get_requests()
        tokens = max_new_tokens or self.max_new_tokens
        chat_mode = kwargs.get("chat_mode") or self.chat_mode
        if chat_mode == "chat":
            payload = {
                "model": kwargs.get("model") or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        else:
            payload = {
                "prompt": prompt,
                "max_tokens": tokens,
                "max_new_tokens": tokens,  # handle services expecting either field
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        # extra_payload allows callers to extend the JSON body when needed
        extra_payload = kwargs.get("extra_payload", {})
        payload.update(extra_payload)

        request_opts = {**self.request_kwargs, **kwargs.get("request_kwargs", {})}
        response = requests.post(
            self.api_url,
            json=payload,
            headers=self.headers,
            **request_opts,
        )
        response.raise_for_status()
        data = response.json()

        text = data.get("text") or data.get("generated_text")
        if text is None and isinstance(data, dict) and data.get("choices"):
            choice = data["choices"][0]
            text = (
                choice.get("message", {}).get("content")
                if isinstance(choice, dict)
                else None
            ) or choice.get("text")
        text = text or ""
        return GenerateResult(text=text, raw=data)


class VLLMOfflineLLM(BaseLLM):
    """Offline vLLM backend."""

    def __init__(self, model_path: str, **kwargs) -> None:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        super().__init__(max_new_tokens, temperature, top_p)

        self.model_path = model_path
        self.tokenizer_name = kwargs.get("tokenizer_name", model_path)
        self.LLM, self.SamplingParams = _get_vllm()
        self.llm = self.LLM(model=model_path, tokenizer=self.tokenizer_name)

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs
    ) -> GenerateResult:
        tokens = max_new_tokens or self.max_new_tokens
        params = self.SamplingParams(
            max_tokens=tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        outputs = self.llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        return GenerateResult(text=text, raw=outputs)

    def predict_options(
        self, prompt: str, options: List[str], **kwargs
    ) -> Dict[str, float]:
        torch = _get_torch()
        params = self.SamplingParams(
            max_tokens=1,
            temperature=0,
            top_p=1.0,
            logprobs=max(len(options), 5),
        )
        outputs = self.llm.generate([prompt], params)
        logprobs = outputs[0].outputs[0].logprobs[0]
        token_logprob = {}
        for token_info in logprobs:
            token_logprob[token_info.token.strip()] = token_info.logprob
        scores = []
        for opt in options:
            scores.append(token_logprob.get(opt, float("-inf")))
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.nn.functional.softmax(scores_tensor, dim=0).cpu().numpy().tolist()
        return {opt: prob for opt, prob in zip(options, probs)}

    def cleanup(self) -> None:
        try:
            torch = _get_torch()
            del self.llm
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Cleanup failed: %s", exc)


def create_llm(config: Dict[str, Any]) -> BaseLLM:
    """Factory to create backend from config."""
    llm_type = config.get("type", config.get("provider", "transformers")).lower()
    model_name = config.get("llm_path") or config.get("llm_name")
    common_kwargs = {
        "max_new_tokens": config.get("max_new_tokens", 1200),
        "temperature": config.get("temperature", 0.7),
        "top_p": config.get("top_p", 0.9),
        "torch_dtype": config.get("torch_dtype"),
        "tokenizer_name": config.get("tokenizer_name", model_name),
        "trust_remote_code": config.get("trust_remote_code", False),
        "model_kwargs": config.get("model_kwargs"),
        "tokenizer_kwargs": config.get("tokenizer_kwargs"),
    }

    if llm_type == "transformers":
        return TransformersLLM(
            model_name=model_name,
            device=config.get("llm_device"),
            **common_kwargs,
        )
    if llm_type in {"openai", "openai_api"}:
        return OpenAIChatLLM(
            model=config.get("llm_name", model_name),
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base"),
            **common_kwargs,
        )
    if llm_type in {"vllm_api", "vllm-openai"}:
        return OpenAIChatLLM(
            model=config.get("llm_name", model_name),
            api_key=config.get("api_key", "EMPTY"),
            api_base=config.get("api_base"),
            **common_kwargs,
        )
    if llm_type in {"requests", "http", "vllm_http"}:
        api_url = config.get("api_url") or config.get("endpoint")
        if not api_url:
            raise ValueError("api_url is required for requests/http backend.")
        return RequestsLLM(
            api_url=api_url,
            headers=config.get("api_headers") or config.get("headers"),
            request_kwargs=config.get("request_kwargs", {}),
            chat_mode=config.get("chat_mode"),
            model=config.get("llm_name") or config.get("model"),
            **common_kwargs,
        )
    if llm_type in {"vllm_offline", "vllm"}:
        return VLLMOfflineLLM(
            model_path=model_name,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported llm type: {llm_type}")

