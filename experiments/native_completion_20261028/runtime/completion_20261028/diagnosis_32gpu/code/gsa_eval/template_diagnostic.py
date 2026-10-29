"""Isolated explicit-template adapter for registered baseline diagnostics.

The formal evaluator remains byte-for-byte frozen. This module is imported only by
``run_benchmark_template_diagnostic.py`` and records its own source fingerprint.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import runner as formal


DIAGNOSTIC_PROTOCOL = "gsa-eval-v7-template-diagnostic-v1"
_SAFE_TEMPLATE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORMAL_ENGINE = formal._engine
_template_type: str | None = None
_model_type: str | None = None


def diagnostic_code_sha256() -> str:
    """Fingerprint the frozen evaluator plus both isolated diagnostic sources."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "gsa_eval/datasets.py",
        "gsa_eval/parsing.py",
        "gsa_eval/registry.py",
        "gsa_eval/runner.py",
        "gsa_eval/scorers.py",
        "run_benchmark.py",
        "gsa_eval/template_diagnostic.py",
        "run_benchmark_template_diagnostic.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _diagnostic_engine(
    model: str,
    max_batch_size: int,
    device_map: str | None = "cuda:0",
    thinking: bool | None = None,
):
    if _template_type is None and _model_type is None:
        return _FORMAL_ENGINE(
            model,
            max_batch_size,
            device_map=device_map,
            thinking=thinking,
        )

    from swift import RequestConfig  # noqa: F401
    from swift.infer_engine import TransformersEngine

    kwargs = {} if device_map is None else {"device_map": device_map}
    if _model_type is not None:
        kwargs["model_type"] = _model_type
    engine = TransformersEngine(
        model,
        max_batch_size=max_batch_size,
        template_type=_template_type,
        **kwargs,
    )
    if thinking is not None:
        tmpl = getattr(engine, "template", None)
        if tmpl is None:
            raise RuntimeError("engine exposes no template; cannot set thinking mode")
        meta = getattr(tmpl, "template_meta", None)
        if thinking:
            prefix = getattr(meta, "thinking_prefix", "") or "<think>\n"
        else:
            prefix = getattr(meta, "non_thinking_prefix", "") or ""
        tmpl.response_prefix = prefix
        tmpl.enable_thinking = thinking
        print(
            f"[diagnostic-engine] template_type={_template_type} "
            f"thinking={thinking} response_prefix={prefix!r}",
            flush=True,
        )
    return engine


def configure_template_type(template_type: str | None, model_type: str | None = None) -> None:
    """Install the process-local engine adapter for one isolated CLI invocation."""
    if template_type is not None and _SAFE_TEMPLATE.fullmatch(template_type) is None:
        raise ValueError(f"unsafe template_type: {template_type!r}")
    if model_type is not None and _SAFE_TEMPLATE.fullmatch(model_type) is None:
        raise ValueError(f"unsafe model_type: {model_type!r}")
    global _template_type, _model_type
    _template_type = template_type
    _model_type = model_type
    formal._ENGINE_CACHE.clear()
    formal._engine = _diagnostic_engine


def _stamp(record: dict) -> dict:
    stamped = dict(record)
    stamped.update(
        {
            "evaluation_protocol": DIAGNOSTIC_PROTOCOL,
            "evaluation_code_sha256": diagnostic_code_sha256(),
            "formal_evaluation_code_sha256": formal.evaluation_code_sha256(),
            "template_type": _template_type,
            "model_type_override": _model_type,
            "diagnostic_isolated": True,
        }
    )
    return stamped


def generate_shard_multi(*args, **kwargs) -> dict:
    return _stamp(formal.generate_shard_multi(*args, **kwargs))


def evaluate(*args, **kwargs) -> dict:
    return _stamp(formal.evaluate(*args, **kwargs))


def score_shards(*args, **kwargs) -> dict:
    return _stamp(formal.score_shards(*args, **kwargs))
