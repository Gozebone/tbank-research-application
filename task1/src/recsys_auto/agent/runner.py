"""ReAct-цикл агента поверх litellm.

Никакого фреймворка оркестрации: один цикл «модель -> tool call -> ответ ->
модель», жёсткий лимит итераций и полный транскрипт в JSON для аудита.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..utils.io import save_json
from .prompts import build_nudge, build_user_prompt, system_prompt
from .tools import AgentTools

DEFAULT_MODEL = "qwen/qwen3.8-omni-flash"


@dataclass
class AgentRun:
    pool: dict
    summary: str | None
    n_iterations: int
    n_propose_calls: int
    n_llm_calls: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    transcript_path: Path | None = None
    finish_reason: str = "budget"
    messages: list[dict] = field(default_factory=list)

    def stats(self) -> dict:
        return {
            "n_iterations": self.n_iterations,
            "n_propose_calls": self.n_propose_calls,
            "n_llm_calls": self.n_llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 1),
            "finish_reason": self.finish_reason,
            "n_pool": len(self.pool),
        }


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _message_dict(message) -> dict:
    """Нормализовать ответ модели в dict, совместимый с litellm."""
    payload: dict = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in tool_calls
        ]
    return payload


def run_agent(
    cfg: Config,
    tools: AgentTools,
    model: str | None = None,
    max_iterations: int | None = None,
    max_propose_calls: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    price_in: float | None = None,
    price_out: float | None = None,
    verbose: bool = True,
) -> AgentRun:
    _load_env()
    import litellm

    agent_cfg = cfg.get("agent", {}) or {}
    model_name = model or agent_cfg.get("model", DEFAULT_MODEL)
    full_model = model_name if "/" in model_name and model_name.startswith("openrouter/") else f"openrouter/{model_name}"
    max_iterations = int(max_iterations or agent_cfg.get("max_iterations", 12))
    max_propose_calls = int(max_propose_calls or agent_cfg.get("max_requests", 60))
    temperature = float(temperature if temperature is not None else agent_cfg.get("temperature", 0.2))
    max_tokens = int(max_tokens or agent_cfg.get("max_tokens", 1500))
    price_in = float(
        price_in if price_in is not None else agent_cfg.get("price_in_per_mtok", 0.0)
    )
    price_out = float(
        price_out if price_out is not None else agent_cfg.get("price_out_per_mtok", 0.0)
    )

    if not os.environ.get("OPENROUTER_API_KEY") and full_model.startswith("openrouter/"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it into .env or export it."
        )

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    import logging as _logging

    for noisy in ("LiteLLM", "litellm"):
        _logging.getLogger(noisy).setLevel(_logging.WARNING)
    tool_schemas = tools.tool_schemas()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(
            tools.batch_size, tools.max_per_entity_metric, extended=tools.extended
        )},
        {
            "role": "user",
            "content": build_user_prompt(
                tools.get_context(),
                budget_left=max_propose_calls,
                batch_size=tools.batch_size,
                extended=tools.extended,
            ),
        },
    ]

    started = time.perf_counter()
    prompt_tokens = completion_tokens = 0
    n_llm_calls = n_propose_calls = 0
    finish_reason = "budget"
    nudges = 0

    for iteration in range(1, max_iterations + 1):
        if n_propose_calls >= max_propose_calls:
            finish_reason = "propose_budget"
            break
        response = litellm.completion(
            model=full_model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=2,
        )
        n_llm_calls += 1
        if response.usage:
            prompt_tokens += int(response.usage.prompt_tokens or 0)
            completion_tokens += int(response.usage.completion_tokens or 0)
        message = response.choices[0].message
        payload = _message_dict(message)
        messages.append(payload)
        tool_calls = payload.get("tool_calls") or []

        if not tool_calls:
            nudges += 1
            if verbose:
                print(f"[{iteration}] model returned text, nudging")
            if nudges > 2:
                finish_reason = "no_tool_calls"
                break
            messages.append(
                {
                    "role": "user",
                    "content": build_nudge(
                        budget_left=max_propose_calls - n_propose_calls,
                        batch_size=tools.batch_size,
                    ),
                }
            )
            continue

        for call in tool_calls:
            name = call["function"]["name"]
            result = tools.dispatch(name, call["function"]["arguments"])
            if name.startswith("propose_"):
                n_propose_calls += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
            if verbose:
                if name.startswith("propose_"):
                    accepted = sum(
                        1 for f in result.get("feedback", []) if f.get("status") == "accepted"
                    )
                    print(
                        f"[{iteration}] {name}: {len(result.get('feedback', []))} specs, "
                        f"{accepted} accepted, pool={result.get('pool_size')}"
                    )
                else:
                    print(f"[{iteration}] {name}")
            if name == "finish":
                finish_reason = "finished"
                break
        if tools.finished:
            finish_reason = "finished"
            break
        if iteration == max_iterations:
            finish_reason = "max_iterations"

    elapsed = time.perf_counter() - started
    cost = prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
    run = AgentRun(
        pool=dict(tools.pool),
        summary=tools.summary,
        n_iterations=len(tools.history),
        n_propose_calls=n_propose_calls,
        n_llm_calls=n_llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        seconds=elapsed,
        transcript_path=tools.transcript_path,
        finish_reason=finish_reason,
        messages=messages,
    )
    save_json(
        {
            "model": full_model,
            "tag": tools.tag,
            "stats": run.stats(),
            "summary": run.summary,
            "pool": list(run.pool.values()),
            "messages": [
                m
                for m in messages
                if m.get("role") != "tool" or len(str(m.get("content", ""))) < 4000
            ],
        },
        tools.transcript_path,
    )
    return run
