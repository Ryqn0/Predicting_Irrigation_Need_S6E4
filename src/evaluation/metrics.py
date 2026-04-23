import numpy as np


def summarize(cv_scores: np.ndarray) -> str:
    return (
        f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )
