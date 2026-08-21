

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        """True when generation stopped because it ran out of output budget.

        This matters more than it looks. A batched plan that is cut in the
        middle produces unparsable JSON and therefore ZERO actions, not a
        partial plan. Russian content needs roughly 1.4x more output tokens
        for the same plan, so the Russian arm reaches this cliff earlier.
        If we do not log it separately we would report it as "the model
        cannot write JSON", which is false and would be the wrong finding.
        """
        return self.finish_reason == "length"


class BaseBackend:
    name: str = "base"

    def generate(self, messages: List[Dict[str, str]]) -> Generation:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------
# oracle backend: replays the gold trace. Used to test the harness itself.
# --------------------------------------------------------------------------

class OracleBackend(BaseBackend):
    """
    Not a model. It replays a known-correct action sequence.

    Purpose: if the oracle does not reach 100% success on every task, then the
    environment or the checks are broken, not the model. Always run this first.
    """

    name = "oracle"

    def __init__(self, gold_actions: List[Dict[str, Any]]):
        self.gold = list(gold_actions)
        self.index = 0

    def reset(self, gold_actions: List[Dict[str, Any]]) -> None:
        self.gold = list(gold_actions)
        self.index = 0

    def generate(self, messages: List[Dict[str, str]]) -> Generation:
        if self.index < len(self.gold):
            action = self.gold[self.index]
            self.index += 1
        else:
            action = {"tool": "done", "args": {}}
        text = json.dumps(action, ensure_ascii=False)
        return Generation(text=text, prompt_tokens=0, completion_tokens=0, latency_s=0.0)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# --------------------------------------------------------------------------
# local backend (transformers + bitsandbytes 4-bit)
# --------------------------------------------------------------------------

class LocalHFBackend(BaseBackend):
    """
    Runs an instruct model locally in 4-bit.

    Tested target: RTX 2060 6 GB. Models that fit comfortably:
        Qwen/Qwen2.5-3B-Instruct
        meta-llama/Llama-3.2-3B-Instruct
    Models that fit only with a short context:
        Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        load_in_4bit: bool = True,
        seed: Optional[int] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        if seed is not None:
            torch.manual_seed(seed)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages: List[Dict[str, str]]) -> Generation:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        start = time.time()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.95 if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - start

        new_ids = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Did generation end on its own, or did it run out of budget?
        # `generate` appends the stop token when it stops naturally, so the
        # presence of any stop id in the tail is the reliable signal.
        stop_ids = {self.tokenizer.eos_token_id}
        extra = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(extra, int):
            stop_ids.add(extra)
        elif isinstance(extra, (list, tuple)):
            stop_ids.update(extra)
        stop_ids.discard(None)

        tail = int(new_ids[-1]) if new_ids.shape[-1] else None
        hit_cap = int(new_ids.shape[-1]) >= self.max_new_tokens
        finish_reason = "length" if (hit_cap and tail not in stop_ids) else "stop"

        return Generation(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_ids.shape[-1]),
            latency_s=latency,
            finish_reason=finish_reason,
        )


# --------------------------------------------------------------------------
# OpenAI-compatible backend (Groq / OpenRouter free tiers, local vLLM, etc.)
# --------------------------------------------------------------------------

class OpenAICompatBackend(BaseBackend):
    """
    Works with any endpoint that speaks the OpenAI chat-completions format.

    Set two environment variables before running:
        ALT_API_BASE  e.g. https://api.groq.com/openai/v1
        ALT_API_KEY   your key
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_pause_s: float = 2.0,
    ):
        import requests  # noqa: F401  (checked early so the error is clear)

        self.requests = __import__("requests")
        self.name = model_id
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.base_url = (base_url or os.environ.get("ALT_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("ALT_API_KEY", "")
        self.request_pause_s = request_pause_s

        if not self.base_url or not self.api_key:
            raise RuntimeError("Set ALT_API_BASE and ALT_API_KEY first.")

    def count_tokens(self, text: str) -> int:
        # Rough fallback only. Real counts come from the API `usage` field.
        return max(1, len(text) // 4)

    def generate(self, messages: List[Dict[str, str]]) -> Generation:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        start = time.time()
        last_error = None
        for attempt in range(5):
            try:
                resp = self.requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                time.sleep(self.request_pause_s)
                return Generation(
                    text=text,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_s=time.time() - start,
                    finish_reason=str(choice.get("finish_reason") or "stop"),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"API request failed after retries: {last_error}")


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def build_backend(kind: str, model_id: str, **kwargs) -> BaseBackend:
    if kind == "local":
        return LocalHFBackend(model_id, **kwargs)
    if kind == "api":
        return OpenAICompatBackend(model_id, **kwargs)
    if kind == "oracle":
        return OracleBackend([])
    raise ValueError(f"unknown backend kind: {kind}")
