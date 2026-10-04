from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiModelConfig:
    provider: str
    name: str
    tier: str
    api_env: str
    input_per_1m: float
    output_per_1m: float


@dataclass(frozen=True)
class ApiUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ApiResponse:
    text: str
    usage: ApiUsage
    latency_ms: float


class ApiBackend:
    def __init__(self, config: ApiModelConfig, timeout_seconds: int = 60):
        self.config = config
        self.timeout_seconds = timeout_seconds

    @property
    def key(self) -> str:
        value = os.environ.get(self.config.api_env)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {self.config.api_env}")
        return value

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> ApiResponse:
        if self.config.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, temperature, max_output_tokens)
        if self.config.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt, temperature, max_output_tokens)
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _generate_openai(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> ApiResponse:
        payload = {
            "model": self.config.name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "trust_decision",
                    "strict": True,
                    "schema": _response_schema(),
                }
            },
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        data = _urlopen_json(request, self.timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = data.get("usage", {})
        text = _extract_openai_text(data)
        return ApiResponse(
            text=text,
            usage=ApiUsage(
                input_tokens=int(usage.get("input_tokens", _rough_tokens(system_prompt + user_prompt))),
                output_tokens=int(usage.get("output_tokens", _rough_tokens(text))),
            ),
            latency_ms=latency_ms,
        )

    def _generate_gemini(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> ApiResponse:
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(),
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.name}:generateContent?key={self.key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        data = _urlopen_json(request, self.timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = data.get("usageMetadata", {})
        text = _extract_gemini_text(data)
        return ApiResponse(
            text=text,
            usage=ApiUsage(
                input_tokens=int(usage.get("promptTokenCount", _rough_tokens(system_prompt + user_prompt))),
                output_tokens=int(usage.get("candidatesTokenCount", _rough_tokens(text))),
            ),
            latency_ms=latency_ms,
        )


def _urlopen_json(request: urllib.request.Request, timeout_seconds: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:800]}") from error


def _extract_openai_text(data: dict[str, Any]) -> str:
    if "output_text" in data:
        return str(data["output_text"])
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    if chunks:
        return "\n".join(chunks)
    return json.dumps(data)


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return json.dumps(data)
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(str(part.get("text", "")) for part in parts if "text" in part)


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["accept", "verify", "abstain", "escalate"]},
            "trust_score": {"type": "number", "minimum": 0, "maximum": 1},
            "used_tool_output": {"type": "boolean"},
            "would_verify": {"type": "boolean"},
            "final_answer": {"type": "string", "maxLength": 180},
            "rationale": {"type": "string", "maxLength": 180},
        },
        "required": ["action", "trust_score", "used_tool_output", "would_verify", "final_answer", "rationale"],
    }


def _gemini_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": ["accept", "verify", "abstain", "escalate"]},
            "trust_score": {"type": "NUMBER"},
            "used_tool_output": {"type": "BOOLEAN"},
            "would_verify": {"type": "BOOLEAN"},
            "final_answer": {"type": "STRING", "maxLength": 180},
            "rationale": {"type": "STRING", "maxLength": 180},
        },
        "required": ["action", "trust_score", "used_tool_output", "would_verify", "final_answer", "rationale"],
    }
