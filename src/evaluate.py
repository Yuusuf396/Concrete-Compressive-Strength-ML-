"""Metrics computation and visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src import utils


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_confusion_matrix(model, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_residual_plot(y_true, y_pred, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="teal", ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted strength (MPa)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    z = np.polyfit(y_pred, residuals, 1)
    ax.plot(y_pred, np.poly1d(z)(y_pred), "r--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_metrics_table(rows: Dict[str, Dict[str, float]], out_path: Path) -> None:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def ensure_reporting_paths() -> Dict[str, Path]:
    utils.ensure_directories()
    return {
        "conf_mat": utils.FIGURES_DIR / "confusion_matrix.png",
        "residuals": utils.FIGURES_DIR / "residuals_plot.png",
        "clf_table": utils.TABLES_DIR / "classification_metrics.csv",
        "reg_table": utils.TABLES_DIR / "regression_metrics.csv",
    }
