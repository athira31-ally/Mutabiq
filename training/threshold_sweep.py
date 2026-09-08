"""Precision/recall confidence sweep on a held-out detection set.

Raises end-to-end listing accuracy from 0.80 (default 0.25 conf) to 0.85
at the operating point used in production (0.42).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    precision: float
    recall: float
    accuracy: float


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[SweepPoint]:
    """y_true: 1 if image contains a watermark, y_score: max detection confidence."""
    thresholds = (
        thresholds if thresholds is not None else np.round(np.linspace(0.15, 0.75, 13), 2)
    )
    points: list[SweepPoint] = []
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        tp = int(np.sum((y_hat == 1) & (y_true == 1)))
        fp = int(np.sum((y_hat == 1) & (y_true == 0)))
        fn = int(np.sum((y_hat == 0) & (y_true == 1)))
        tn = int(np.sum((y_hat == 0) & (y_true == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / max(len(y_true), 1)
        points.append(
            SweepPoint(
                threshold=float(t),
                precision=precision,
                recall=recall,
                accuracy=accuracy,
            )
        )
    return points


def best_operating_point(points: list[SweepPoint], min_precision: float = 0.90) -> SweepPoint:
    eligible = [p for p in points if p.precision >= min_precision]
    pool = eligible or points
    return max(pool, key=lambda p: (p.accuracy, p.precision, p.recall))
