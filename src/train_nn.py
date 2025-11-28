"""Neural-network style baselines using scikit-learn MLP models."""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn

from sklearn.neural_network import MLPClassifier, MLPRegressor

from src import evaluate, utils
from src.data import SplitBundle, add_targets, build_splits, load_raw_dataframe
from src.features import inject_features, make_feature_frame, scale_features

EXPERIMENT_NAME = "ConcreteStrength_NeuralNets"


def main() -> None:
    utils.ensure_directories()
    utils.set_all_seeds(21)
    mlflow.set_tracking_uri(f"file:{utils.MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_dataframe()
    df, _ = add_targets(df)
    feature_frame = make_feature_frame(df)
    splits = build_splits(df)

    clf_bundle = inject_features(splits["classification"], feature_frame)
    reg_bundle = inject_features(splits["regression"], feature_frame)

    clf_features = scale_features(clf_bundle)
    reg_features = scale_features(reg_bundle)

    clf_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=21,
        early_stopping=True,
    )

    reg_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=21,
        early_stopping=True,
    )

    with mlflow.start_run(run_name="nn_classifier"):
        clf_model.fit(clf_features.X_train, clf_bundle.y_train)
        val_pred = clf_model.predict(clf_features.X_val)
        test_pred = clf_model.predict(clf_features.X_test)

        val_metrics = evaluate.classification_metrics(clf_bundle.y_val, val_pred)
        test_metrics = evaluate.classification_metrics(clf_bundle.y_test, test_pred)

        mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric("val_f1", val_metrics["f1"])
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_f1", test_metrics["f1"])

        evaluate.save_confusion_matrix(
            clf_model,
            clf_features.X_test,
            clf_bundle.y_test,
            utils.FIGURES_DIR / "nn_confusion_matrix.png",
        )
        model_path = utils.MODELS_DIR / "mlp_classifier.joblib"
        joblib.dump(clf_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

    with mlflow.start_run(run_name="nn_regressor"):
        reg_model.fit(reg_features.X_train, reg_bundle.y_train)
        val_pred = reg_model.predict(reg_features.X_val)
        test_pred = reg_model.predict(reg_features.X_test)

        val_metrics = evaluate.regression_metrics(reg_bundle.y_val, val_pred)
        test_metrics = evaluate.regression_metrics(reg_bundle.y_test, test_pred)

        mlflow.log_metric("val_mae", val_metrics["mae"])
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_rmse", test_metrics["rmse"])

        evaluate.save_residual_plot(
            reg_bundle.y_test,
            test_pred,
            utils.FIGURES_DIR / "nn_residuals.png",
        )
        model_path = utils.MODELS_DIR / "mlp_regressor.joblib"
        joblib.dump(reg_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

    print("Neural network training complete.")


if __name__ == "__main__":
    main()
