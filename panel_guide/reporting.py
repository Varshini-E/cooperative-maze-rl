"""Shared run metadata and usage accounting, independent of CLI entry points."""

import importlib.metadata
import json
from pathlib import Path


def dependency_versions() -> dict:
    versions = {}
    for name in ("panel-guide", "pillow", "tinker", "tinker-cookbook"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def usage(directory: Path) -> dict:
    """Count returned generations once, across native and conversation runners."""
    totals = {
        role: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_uncached_usd": 0.0}
        for role in ("expert", "operator")
    }
    for path in directory.rglob("trace.jsonl"):
        with path.open() as trace:
            for line in trace:
                event = json.loads(line)
                sample = event.get("sample", {})
                # Native policy samples carry usage; their parsed completions do not.
                if event["type"] not in ("completion", "policy_sample") or "input_tokens" not in sample:
                    continue
                role = totals[event["role"]]
                role["calls"] += 1
                for key in ("input_tokens", "output_tokens"):
                    role[key] += sample[key]
                prices = sample["config"]
                role["estimated_uncached_usd"] += (
                    sample["input_tokens"] * prices["input_usd_per_million"]
                    + sample["output_tokens"] * prices["output_usd_per_million"]
                ) / 1_000_000
    return totals
