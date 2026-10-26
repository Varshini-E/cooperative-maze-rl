"""Optional native Tinker inference. No training client or optimizer lives here."""

import io
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from panel_guide.conversation import BudgetExceeded, Completion, Request


@dataclass(frozen=True)
class ModelConfig:
    model: str
    renderer: str
    max_tokens: int
    context_tokens: int
    temperature: float
    input_usd_per_million: float
    output_usd_per_million: float

    def __post_init__(self) -> None:
        for count in (self.max_tokens, self.context_tokens):
            if type(count) is not int or count <= 0:
                raise ValueError("Token limits must be positive integers.")
        for value in (self.temperature, self.input_usd_per_million, self.output_usd_per_million):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Temperature and prices must be finite and nonnegative.")
        if self.input_usd_per_million == 0 or self.output_usd_per_million == 0:
            raise ValueError("Configure nonzero prices before paid inference.")


class SpendingGuard:
    """Conservative per-run estimate, not a provider-enforced billing limit.

    Reserve uncached prompt cost plus the maximum output cost, with 20% headroom.
    Keep reservations after failures and short outputs. Never credit uncertain calls.
    """

    def __init__(self, cap_usd: float, record: Callable[..., None] | None = None) -> None:
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("A positive finite --budget-usd is required for Tinker.")
        self.cap_usd = cap_usd
        self.reserved_usd = 0.0
        self.reservations: list[dict] = []
        self.record = record

    def reserve(self, role: str, config: ModelConfig, prompt_tokens: int) -> float:
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("Cannot reserve a call without its rendered input length, including images.")
        amount = 1.2 * (prompt_tokens * config.input_usd_per_million
                        + config.max_tokens * config.output_usd_per_million) / 1_000_000
        if self.reserved_usd + amount > self.cap_usd:
            raise BudgetExceeded("The next call would exceed the local spending estimate.")
        self.reserved_usd += amount
        event = {"role": role, "model": config.model, "prompt_tokens": prompt_tokens,
                 "max_output_tokens": config.max_tokens, "reserved_usd": amount,
                 "total_reserved_usd": self.reserved_usd}
        self.reservations.append(event)
        if self.record:
            self.record(type="cost_reservation", **event)
        return amount

    def to_dict(self) -> dict:
        return {"cap_usd": self.cap_usd, "reserved_usd": self.reserved_usd,
                "note": "Conservative estimate, not an invoice or provider spending limit.",
                "reservations": self.reservations}

    def reserve_training(self, model: str, tokens: int, usd_per_million: float, *, operation: str) -> float:
        """Training-client forward passes bill all input tokens, including context."""
        if type(tokens) is not int or tokens <= 0 or not math.isfinite(usd_per_million) or usd_per_million <= 0:
            raise ValueError('Training reservations need positive token counts and prices.')
        amount = 1.2 * tokens * usd_per_million / 1_000_000
        if self.reserved_usd + amount > self.cap_usd:
            raise BudgetExceeded('The next training operation would exceed the local spending estimate.')
        self.reserved_usd += amount
        event = {'role': 'training', 'model': model, 'operation': operation, 'tokens': tokens,
                 'reserved_usd': amount, 'total_reserved_usd': self.reserved_usd}
        self.reservations.append(event)
        if self.record:
            self.record(type='cost_reservation', **event)
        return amount


def model_messages(request: Request) -> list[dict]:
    messages = request.text_messages()
    if request.role == "expert":
        if request.image_png is not None or request.detail_png is not None:
            raise ValueError("Expert inputs must be text-only.")
    elif request.role == "operator":
        if request.image_png is None:
            raise ValueError("Operator needs the current screenshot.")
        with Image.open(io.BytesIO(request.image_png)) as image:
            current = image.convert("RGB")
        parts = [{"type": "image", "image": current}]
        if request.detail_png is not None:
            with Image.open(io.BytesIO(request.detail_png)) as image:
                detail = image.convert("RGB")
            parts = [{"type": "text", "text": "Image 1: whole maze (current overview)."}, *parts,
                     {"type": "text", "text": "Image 2: enlarged current-room crop from that same overview."},
                     {"type": "image", "image": detail}]
        messages[-1]["content"] = [*parts, {"type": "text", "text": messages[-1]["content"]}]
    else:
        raise ValueError("Unknown agent role.")
    return messages


def public_text(content: Any) -> str:
    """Only final text crosses to the partner, never renderer thinking parts."""
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") == "text")


class TinkerAgent:
    def __init__(self, config: ModelConfig, renderer: Any, sampler: Any,
                 budget: SpendingGuard, sampling_params: Callable[..., Any], *,
                 capture_dir: Path | None = None, decode_tokens: Callable | None = None) -> None:
        self.config = config
        self.renderer = renderer
        self.sampler = sampler
        self.budget = budget
        self.sampling_params = sampling_params
        self.capture_dir = capture_dir
        self.decode_tokens = decode_tokens
        self.call_count = 0
        self.policy_id = f'base:{config.model}'
        self.model_path = None

    async def sample_tokens(self, prompt: Any, *, role: str, stop=None, max_tokens: int | None = None):
        """Sample once with the same capture/budget safeguards for either runner."""
        if role not in ("expert", "operator"):
            raise ValueError("Unknown agent role.")
        config = self.config
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens <= 0:
                raise ValueError("max_tokens must be positive.")
            config = replace(config, max_tokens=min(max_tokens, config.max_tokens))
        # Expert trajectories must be text-only, including native Env callers.
        prompt_ids = prompt.to_ints() if role == "expert" else None
        prompt_length = prompt.length
        if prompt_length + config.max_tokens > config.context_tokens:
            raise ValueError("Rendered request exceeds the configured model context.")
        reservation = self.budget.reserve(role, config, prompt_length)
        params = self.sampling_params(
            max_tokens=config.max_tokens, temperature=config.temperature,
            stop=self.renderer.get_stop_sequences() if stop is None else stop,
        )
        self.call_count += 1
        capture_path = None
        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = self.capture_dir / f"{role}-{self.call_count:04d}.json"
            with capture_path.open("x") as stream:
                json.dump({"model": self.config.model, "num_samples": 1,
                           "policy_id": self.policy_id, "model_path": self.model_path,
                           "prompt": prompt.model_dump(mode="json"),
                           "sampling_params": params.model_dump(mode="json")}, stream)
        result = await self.sampler.sample_async(
            prompt=prompt, num_samples=1,
            sampling_params=params,
        )
        sequence = result.sequences[0]
        # Preserve original sampled tokens, including any private reasoning, for audit.
        # These records do not yet constitute a verified RL training batch.
        metadata = {
            "policy_id": self.policy_id, "model_path": self.model_path,
            "config": asdict(config), "input_tokens": prompt_length,
            "output_tokens": len(sequence.tokens), "token_ids": list(sequence.tokens),
            "logprobs": None if sequence.logprobs is None else list(sequence.logprobs),
            "reserved_usd": reservation,
            "provider_request": str(capture_path) if capture_path else None,
            "raw_text": self.decode_tokens(sequence.tokens) if self.decode_tokens else None,
        }
        if prompt_ids is not None:
            metadata["prompt_token_ids"] = prompt_ids
        return sequence, metadata

    async def complete(self, request: Request) -> Completion:
        prompt = self.renderer.build_generation_prompt(model_messages(request))
        sequence, metadata = await self.sample_tokens(prompt, role=request.role)
        try:
            parsed, termination = self.renderer.parse_response(sequence.tokens)
        except ValueError as exc:
            metadata["parse_error_class"] = type(exc).__name__
            return Completion("", "invalid_termination", metadata)
        metadata["parse_termination"] = str(termination)
        reason = "stop" if sequence.stop_reason == "stop" and termination.is_clean else "invalid_termination"
        return Completion(public_text(parsed["content"]), reason, metadata)


async def create_agents(expert: ModelConfig, operator: ModelConfig, budget: SpendingGuard,
                        *, capture_dir: Path | None = None):
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("Set TINKER_API_KEY locally before selecting the Tinker backend.")
    try:
        import tinker
        from tinker.lib.retry_handler import RetryConfig
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise RuntimeError("Install model dependencies with: uv sync --locked --extra models") from exc

    service = tinker.ServiceClient(display_name="cooperative-maze-pilot")
    try:
        capabilities = await service.get_server_capabilities_async()
        supported = {model.model_name: model for model in capabilities.supported_models}
        missing = {expert.model, operator.model} - supported.keys()
        if missing:
            raise ValueError(f"Models unavailable to this account: {', '.join(sorted(missing))}")
        agents = []
        for role, config in (("expert", expert), ("operator", operator)):
            access = supported[config.model]
            if access.sampleable is False:
                raise ValueError(f"Sampling is unavailable for {config.model}.")
            if access.max_context_length and config.context_tokens > access.max_context_length:
                raise ValueError(f"Configured context exceeds the account limit for {config.model}.")
            tokenizer = get_tokenizer(config.model)
            processor = get_image_processor(config.model) if role == "operator" else None
            renderer = get_renderer(config.renderer, tokenizer, image_processor=processor)
            sampler = await service.create_sampling_client_async(
                base_model=config.model, retry_config=RetryConfig(enable_retry_logic=False),
            )
            agents.append(TinkerAgent(config, renderer, sampler, budget, tinker.SamplingParams,
                                     capture_dir=capture_dir, decode_tokens=tokenizer.decode))
    except BaseException:
        await service.close("errored")
        raise
    return agents[0], agents[1], service
