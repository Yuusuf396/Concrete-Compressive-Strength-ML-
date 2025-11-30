"""Classical ML baselines for both classification and regression tasks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from src import evaluate, utils
from src.data import add_targets, build_splits, load_raw_dataframe
from src.features import inject_features, make_feature_frame, scale_features

EXPERIMENT_NAME = "ConcreteStrength_Baselines"


def main() -> None:
    utils.ensure_directories()
    utils.set_all_seeds(42)
    mlflow.set_tracking_uri(f"file:{utils.MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_dataframe()
    df, cutoff = add_targets(df)
    feature_frame = make_feature_frame(df)
    splits = build_splits(df)

    clf_bundle = inject_features(splits["classification"], feature_frame)
    reg_bundle = inject_features(splits["regression"], feature_frame)

    clf_features = scale_features(clf_bundle)
    reg_features = scale_features(reg_bundle)

    classification_models = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42, penalty="none"),
        ),
        (
            "LogisticRegression_L2",
            LogisticRegression(
                max_iter=1000, random_state=42, penalty="l2", C=1.0),
        ),
        (
            "LogisticRegression_L1",
            LogisticRegression(max_iter=1000, random_state=42,
                               penalty="l1", solver="saga", C=1.0),
        ),
        (
            "GaussianNB",
            GaussianNB(),
        ),
    ]

    regression_models = [
        ("LinearRegression", LinearRegression()),
        (
            "Ridge_alpha0.1",
            Ridge(alpha=0.1, random_state=42),
        ),
        (
            "Ridge_alpha1.0",
            Ridge(alpha=1.0, random_state=42),
        ),
        (
            "Ridge_alpha10",
            Ridge(alpha=10.0, random_state=42),
        ),
        (
            "Lasso_alpha0.1",
            Lasso(alpha=0.1, random_state=42, max_iter=2000),
        ),
        (
            "ElasticNet",
            ElasticNet(alpha=1.0, l1_ratio=0.5,
                       random_state=42, max_iter=2000),
        ),
        (
            "DecisionTreeRegressor",
            DecisionTreeRegressor(max_depth=6, random_state=42),
        ),
    ]

    classification_results = {}
    regression_results = {}
    best_clf = {"name": None, "model": None, "val_f1": -np.inf}
    best_reg = {"name": None, "model": None, "val_rmse": np.inf}

    for name, model in classification_models:
        with mlflow.start_run(run_name=f"baseline_clf_{name}"):
            model.fit(clf_features.X_train, clf_bundle.y_train)
            val_pred = model.predict(clf_features.X_val)
            test_pred = model.predict(clf_features.X_test)

            val_metrics = evaluate.classification_metrics(
                clf_bundle.y_val, val_pred)
            test_metrics = evaluate.classification_metrics(
                clf_bundle.y_test, test_pred)

            mlflow.log_param("model", name)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
            mlflow.log_metric("val_f1", val_metrics["f1"])
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
            mlflow.log_metric("test_f1", test_metrics["f1"])

            temp_dir = Path(tempfile.mkdtemp())
            conf_path = temp_dir / f"{name}_confusion.png"
            evaluate.save_confusion_matrix(
                model, clf_features.X_test, clf_bundle.y_test, conf_path)
            mlflow.log_artifact(conf_path, artifact_path="figures")

            model_path = utils.MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            classification_results[name] = {
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
            }

            if val_metrics["f1"] > best_clf["val_f1"]:
                best_clf = {"name": name, "model": model,
                            "val_f1": val_metrics["f1"]}

    for name, model in regression_models:
        with mlflow.start_run(run_name=f"baseline_reg_{name}"):
            model.fit(reg_features.X_train, reg_bundle.y_train)
            val_pred = model.predict(reg_features.X_val)
            test_pred = model.predict(reg_features.X_test)

            val_metrics = evaluate.regression_metrics(
                reg_bundle.y_val, val_pred)
            test_metrics = evaluate.regression_metrics(
                reg_bundle.y_test, test_pred)

            mlflow.log_param("model", name)
            mlflow.log_metric("val_mae", val_metrics["mae"])
            mlflow.log_metric("val_rmse", val_metrics["rmse"])
            mlflow.log_metric("test_mae", test_metrics["mae"])
            mlflow.log_metric("test_rmse", test_metrics["rmse"])

            temp_dir = Path(tempfile.mkdtemp())
            residual_path = temp_dir / f"{name}_residuals.png"
            evaluate.save_residual_plot(
                reg_bundle.y_test, test_pred, residual_path)
            mlflow.log_artifact(residual_path, artifact_path="figures")

            model_path = utils.MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            regression_results[name] = {
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }

            if val_metrics["rmse"] < best_reg["val_rmse"]:
                best_reg = {"name": name, "model": model,
                            "val_rmse": val_metrics["rmse"]}

    reporting_paths = evaluate.ensure_reporting_paths()
    evaluate.save_metrics_table(
        classification_results, reporting_paths["clf_table"])
    evaluate.save_metrics_table(
        regression_results, reporting_paths["reg_table"])

    if best_clf["model"] is not None:
        evaluate.save_confusion_matrix(
            best_clf["model"],
            clf_features.X_test,
            clf_bundle.y_test,
            utils.FIGURES_DIR / "best_classification_confusion.png",
        )

    if best_reg["model"] is not None:
        test_pred = best_reg["model"].predict(reg_features.X_test)
        evaluate.save_residual_plot(
            reg_bundle.y_test,
            test_pred,
            utils.FIGURES_DIR / "best_regression_residuals.png",
        )

    print("Baselines complete.")


if __name__ == "__main__":
    main()
