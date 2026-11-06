"""Shared uncertainty metrics for model outputs.

The public helpers return a normalized uncertainty score in [0, 1] and the raw
natural-log entropy. Normalized scores are easier to compare in API responses,
while raw entropy is useful when a model has calibrated thresholds from
validation data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

_EPS = 1e-12


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def categorical_uncertainty(
    probabilities: Iterable[float] | np.ndarray,
) -> dict[str, float | str]:
    """Return normalized Shannon entropy for one mutually-exclusive distribution."""

    probs = _as_float_array(probabilities).reshape(-1)
    if probs.size <= 1:
        return {
            "uncertainty": 0.0,
            "entropy": 0.0,
            "max_entropy": 0.0,
            "method": "categorical_entropy",
        }

    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        probs = np.full(probs.shape, 1.0 / probs.size, dtype=np.float64)
    else:
        probs = probs / total

    entropy = -float(np.sum(probs * np.log(np.clip(probs, _EPS, 1.0))))
    max_entropy = math.log(probs.size)
    uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "uncertainty": _round_metric(np.clip(uncertainty, 0.0, 1.0)),
        "entropy": _round_metric(entropy),
        "max_entropy": _round_metric(max_entropy),
        "method": "categorical_entropy",
    }


def multilabel_uncertainty(
    probabilities: Iterable[float] | np.ndarray,
) -> dict[str, float | str]:
    """Return mean normalized Bernoulli entropy for independent labels."""

    probs = np.clip(_as_float_array(probabilities).reshape(-1), 0.0, 1.0)
    if probs.size == 0:
        return {
            "uncertainty": 1.0,
            "entropy": 0.0,
            "max_entropy": 0.0,
            "method": "mean_binary_entropy",
        }

    safe_probs = np.clip(probs, _EPS, 1.0 - _EPS)
    entropy_per_label = -(
        safe_probs * np.log(safe_probs) + (1.0 - safe_probs) * np.log(1.0 - safe_probs)
    )
    entropy = float(np.mean(entropy_per_label))
    max_entropy = math.log(2.0)
    uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "uncertainty": _round_metric(np.clip(uncertainty, 0.0, 1.0)),
        "entropy": _round_metric(entropy),
        "max_entropy": _round_metric(max_entropy),
        "method": "mean_binary_entropy",
    }


def sigmoid_top_score_uncertainty(
    probabilities: Iterable[float] | np.ndarray,
) -> dict[str, float | str]:
    """Estimate uncertainty for a mutually ranked Sigmoid risk head.

    The legacy glaucoma model exposes independent Sigmoid scores rather than a
    softmax distribution. Use the winning score's binary entropy and preserve
    a competing score above 0.5 as an explicit conflict signal.
    """

    probs = np.clip(_as_float_array(probabilities).reshape(-1), 0.0, 1.0)
    if probs.size == 0:
        return {
            "uncertainty": 1.0,
            "entropy": 0.0,
            "max_entropy": 0.0,
            "method": "sigmoid_top_score_conflict",
        }

    ranked = np.sort(probs)[::-1]
    top = float(np.clip(ranked[0], _EPS, 1.0 - _EPS))
    top_entropy = -(
        top * math.log(top) + (1.0 - top) * math.log(1.0 - top)
    ) / math.log(2.0)
    competing_score = float(ranked[1]) if ranked.size > 1 else 0.0
    uncertainty = max(float(top_entropy), competing_score)
    return {
        "uncertainty": _round_metric(np.clip(uncertainty, 0.0, 1.0)),
        "entropy": _round_metric(top_entropy * math.log(2.0)),
        "max_entropy": _round_metric(math.log(2.0)),
        "method": "sigmoid_top_score_conflict",
    }


def segmentation_uncertainty(probabilities: np.ndarray) -> dict[str, float | str]:
    """Summarize pixel-wise entropy for a segmentation probability map.

    Expected shape is ``(classes, height, width[, depth])``. Single-channel maps
    are treated as foreground Bernoulli probabilities.
    """

    probs = _as_float_array(probabilities)
    if probs.ndim < 2:
        raise ValueError(
            "segmentation probabilities must include class and spatial axes"
        )

    channels = probs.shape[0]
    if channels <= 1:
        values = np.clip(probs[0] if probs.size else probs, 0.0, 1.0)
        safe_values = np.clip(values, _EPS, 1.0 - _EPS)
        entropy_map = -(
            safe_values * np.log(safe_values)
            + (1.0 - safe_values) * np.log(1.0 - safe_values)
        ) / math.log(2.0)
        method = "pixel_binary_entropy"
    else:
        probs = np.clip(probs, 0.0, None)
        totals = probs.sum(axis=0, keepdims=True)
        probs = np.divide(
            probs,
            totals,
            out=np.full_like(probs, 1.0 / channels, dtype=np.float64),
            where=totals > 0,
        )
        entropy_map = -np.sum(probs * np.log(np.clip(probs, _EPS, 1.0)), axis=0)
        entropy_map = entropy_map / math.log(channels)
        method = "pixel_categorical_entropy"

    entropy_map = np.clip(entropy_map, 0.0, 1.0)
    return {
        "uncertainty": _round_metric(float(np.mean(entropy_map))),
        "max_uncertainty": _round_metric(float(np.max(entropy_map))),
        "p95_uncertainty": _round_metric(float(np.percentile(entropy_map, 95))),
        "method": method,
    }


def normalized_threshold_from_entropy(
    entropy_threshold: float, class_count: int
) -> float:
    """Convert a raw Shannon entropy threshold to the normalized [0, 1] scale."""

    if class_count <= 1:
        return 0.0
    return _round_metric(float(entropy_threshold) / math.log(class_count))


def unavailable_uncertainty(reason: str) -> dict[str, Any]:
    return {
        "uncertainty": None,
        "uncertainty_method": "unavailable",
        "uncertainty_reason": reason,
    }
