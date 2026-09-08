import numpy as np

from training.threshold_sweep import best_operating_point, sweep_thresholds


def test_sweep_selects_threshold_improving_accuracy() -> None:
    # 20 held-out images: default 0.25 conf is noisy (80%); 0.42 suppresses FPs (85%).
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array(
        [
            0.95,
            0.90,
            0.88,
            0.70,
            0.60,
            0.55,
            0.50,
            0.20,
            0.50,
            0.48,
            0.30,
            0.10,
            0.08,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
            0.01,
        ]
    )
    points = sweep_thresholds(y_true, y_score, thresholds=np.array([0.25, 0.42]))
    by_t = {round(p.threshold, 2): p for p in points}
    assert round(by_t[0.25].accuracy, 2) == 0.80
    assert round(by_t[0.42].accuracy, 2) == 0.85
    chosen = best_operating_point(points, min_precision=0.70)
    assert chosen.threshold == 0.42
    assert chosen.accuracy > by_t[0.25].accuracy
