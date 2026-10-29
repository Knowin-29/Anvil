"""Self-contained evaluation harness for GSA-Forge.

Replaces the nine external runners under benchmark_eval_all/src, which could not be
driven from one place: their argparse surfaces disagreed (--model vs --model-path vs
no argparse at all), three of them loaded Qwen3.5 checkpoints through an InternVL
AutoModel path that left the language model randomly initialised, and two were API
clients needing a served endpoint. Everything here loads the dataset, runs inference
through swift's TransformersEngine and scores in-process, so a benchmark either
produces a trusted number or reports why it could not.
"""
from .registry import BENCHMARKS, get_spec
from .runner import evaluate, score_shards

__all__ = ["BENCHMARKS", "get_spec", "evaluate", "score_shards"]
