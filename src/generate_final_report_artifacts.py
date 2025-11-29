"""Generate all required artifacts for the Final Report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import joblib

from src import evaluate, utils
from src.data import add_targets, build_splits, load_raw_dataframe
from src.features import inject_features, make_feature_frame, scale_features


def plot_learning_curves(clf_model, reg_model, output_dir: Path):
    """Generate learning curve plots for both NN models."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Classification learning curve
    if hasattr(clf_model, 'loss_curve_'):
        epochs = range(1, len(clf_model.loss_curve_) + 1)
        ax1.plot(epochs, clf_model.loss_curve_, 'b-',
                 linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Classification Neural Network Learning Curve',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

    # Regression learning curve
    if hasattr(reg_model, 'loss_curve_'):
        epochs = range(1, len(reg_model.loss_curve_) + 1)
        ax2.plot(epochs, reg_model.loss_curve_, 'r-',
                 linewidth=2, label='Training Loss')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Regression Neural Network Learning Curve',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved learning curves to {output_path}")

    # Save individual plots for the report
    # Plot 1: Classification learning curve
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    if hasattr(clf_model, 'loss_curve_'):
        epochs = range(1, len(clf_model.loss_curve_) + 1)
        ax1.plot(epochs, clf_model.loss_curve_, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Classification Neural Network Learning Curve',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path1 = output_dir / "plot1_classification_learning_curve.png"
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Plot 1 to {output_path1}")

    # Plot 2: Regression learning curve
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if hasattr(reg_model, 'loss_curve_'):
        epochs = range(1, len(reg_model.loss_curve_) + 1)
        ax2.plot(epochs, reg_model.loss_curve_, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Regression Neural Network Learning Curve',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path2 = output_dir / "plot2_regression_learning_curve.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Plot 2 to {output_path2}")


def plot_feature_importance(reg_model, feature_names, X_test, y_test, output_dir: Path):
    """Generate feature importance plot using permutation importance."""

    print("Computing feature importance (this may take a minute)...")
    perm_importance = permutation_importance(
        reg_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    y_pos = np.arange(len(feature_names))
    importances = perm_importance.importances_mean[sorted_idx]
    std = perm_importance.importances_std[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh(y_pos, importances, xerr=std,
                   color=colors, alpha=0.8, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance (Best Regression Model)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "plot5_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Plot 5 (Feature Importance) to {output_path}")

    # Also save as CSV
    importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance': importances,
        'Std': std
    })
    csv_path = utils.TABLES_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"✓ Saved feature importance data to {csv_path}")


def create_comparison_tables(output_dir: Path):
    """Create comparison tables: Classical vs Neural Network."""

    # Read existing metrics
    clf_classical = pd.read_csv(
        utils.TABLES_DIR / "classification_metrics.csv")
    clf_nn = pd.read_csv(utils.TABLES_DIR / "nn_classification_metrics.csv")

    reg_classical = pd.read_csv(utils.TABLES_DIR / "regression_metrics.csv")
    reg_nn = pd.read_csv(utils.TABLES_DIR / "nn_regression_metrics.csv")

    # Table 1: Classification Comparison
    clf_best_classical = clf_classical.sort_values(
        'test_f1', ascending=False).iloc[0]
    clf_nn_metrics = clf_nn.iloc[0]

    table1 = pd.DataFrame({
        'Model': [clf_best_classical['model'], 'MLPClassifier (Neural Network)'],
        'Validation Accuracy': [
            f"{clf_best_classical['val_accuracy']:.4f}",
            f"{clf_nn_metrics['val_accuracy']:.4f}"
        ],
        'Test Accuracy': [
            f"{clf_best_classical['test_accuracy']:.4f}",
            f"{clf_nn_metrics['test_accuracy']:.4f}"
        ],
        'Validation F1-Score': [
            f"{clf_best_classical['val_f1']:.4f}",
            f"{clf_nn_metrics['val_f1']:.4f}"
        ],
        'Test F1-Score': [
            f"{clf_best_classical['test_f1']:.4f}",
            f"{clf_nn_metrics['test_f1']:.4f}"
        ]
    })

    table1_path = output_dir / "table1_classification_comparison.csv"
    table1.to_csv(table1_path, index=False)
    print(f"✓ Saved Table 1 (Classification Comparison) to {table1_path}")

    # Table 2: Regression Comparison
    reg_best_classical = reg_classical.sort_values(
        'test_rmse', ascending=True).iloc[0]
    reg_nn_metrics = reg_nn.iloc[0]

    table2 = pd.DataFrame({
        'Model': [reg_best_classical['model'], 'MLPRegressor (Neural Network)'],
        'Validation MAE': [
            f"{reg_best_classical['val_mae']:.4f}",
            f"{reg_nn_metrics['val_mae']:.4f}"
        ],
        'Test MAE': [
            f"{reg_best_classical['test_mae']:.4f}",
            f"{reg_nn_metrics['test_mae']:.4f}"
        ],
        'Validation RMSE': [
            f"{reg_best_classical['val_rmse']:.4f}",
            f"{reg_nn_metrics['val_rmse']:.4f}"
        ],
        'Test RMSE': [
            f"{reg_best_classical['test_rmse']:.4f}",
            f"{reg_nn_metrics['test_rmse']:.4f}"
        ]
    })

    table2_path = output_dir / "table2_regression_comparison.csv"
    table2.to_csv(table2_path, index=False)
    print(f"✓ Saved Table 2 (Regression Comparison) to {table2_path}")

    print("\n" + "="*60)
    print("TABLE 1: CLASSIFICATION COMPARISON (Best Classical vs NN)")
    print("="*60)
    print(table1.to_string(index=False))

    print("\n" + "="*60)
    print("TABLE 2: REGRESSION COMPARISON (Best Classical vs NN)")
    print("="*60)
    print(table2.to_string(index=False))


def copy_required_plots():
    """Copy and rename existing plots to match report requirements."""

    figures_dir = utils.FIGURES_DIR

    # Plot 3: Best classification confusion matrix (already exists)
    src = figures_dir / "best_classification_confusion.png"
    dst = figures_dir / "plot3_classification_confusion_matrix.png"
    if src.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"✓ Created Plot 3 (Confusion Matrix): {dst}")

    # Plot 4: Best regression residuals (already exists)
    src = figures_dir / "best_regression_residuals.png"
    dst = figures_dir / "plot4_regression_residuals.png"
    if src.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"✓ Created Plot 4 (Residuals): {dst}")


def create_summary_document():
    """Create a summary document listing all required artifacts."""

    summary = """
FINAL REPORT ARTIFACTS CHECKLIST
=================================

Required Plots (5 total):
-------------------------
✓ Plot 1: reports/figures/plot1_classification_learning_curve.png
✓ Plot 2: reports/figures/plot2_regression_learning_curve.png
✓ Plot 3: reports/figures/plot3_classification_confusion_matrix.png
✓ Plot 4: reports/figures/plot4_regression_residuals.png
✓ Plot 5: reports/figures/plot5_feature_importance.png

Required Tables (2 total):
--------------------------
✓ Table 1: reports/tables/table1_classification_comparison.csv
✓ Table 2: reports/tables/table2_regression_comparison.csv

Additional Resources:
--------------------
- Feature importance data: reports/tables/feature_importance.csv
- All trained models: models/
- Original metrics: reports/tables/

Report Structure Requirements:
------------------------------
1. NN architecture + preprocessing
   - MLP Architecture: Classification (64→32), Regression (128→64)
   - Preprocessing: StandardScaler, engineered features (water/cement ratio, etc.)
   
2. Hyperparameter tuning
   - Learning rate, hidden layer sizes, max iterations
   - Early stopping to prevent overfitting
   
3. Comparative analysis (classical vs NN)
   - See Table 1 and Table 2
   
4. Improvement analysis from Midpoint → Final
   - Document your improvements and iterations
   
5. Risks, ethics, and limitations
   - Data limitations, model assumptions, real-world applicability

Poster Requirements (1 page, landscape):
----------------------------------------
- 2 visuals: Suggest confusion matrix + feature importance
- 1 table: Use combined results from Tables 1 & 2
- Include: Title, dataset description, methods, conclusions

File Naming:
-----------
- final_report_GroupName.pdf
- poster_GroupName.pdf
"""

    summary_path = utils.REPORTS_DIR / "FINAL_REPORT_CHECKLIST.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n✓ Created summary checklist: {summary_path}")


def main():
    """Generate all final report artifacts."""

    print("="*70)
    print("GENERATING FINAL REPORT ARTIFACTS")
    print("="*70)

    utils.ensure_directories()

    # Load data and splits
    print("\n1. Loading data and trained models...")
    df = load_raw_dataframe()
    df, _ = add_targets(df)
    feature_frame = make_feature_frame(df)
    splits = build_splits(df)

    clf_bundle = inject_features(splits["classification"], feature_frame)
    reg_bundle = inject_features(splits["regression"], feature_frame)

    clf_features = scale_features(clf_bundle)
    reg_features = scale_features(reg_bundle)

    # Load trained models
    clf_model = joblib.load(utils.MODELS_DIR / "mlp_classifier.joblib")
    reg_model = joblib.load(utils.MODELS_DIR / "mlp_regressor.joblib")

    print("✓ Data and models loaded")

    # Generate artifacts
    print("\n2. Generating learning curves (Plots 1 & 2)...")
    plot_learning_curves(clf_model, reg_model, utils.FIGURES_DIR)

    print("\n3. Copying and organizing required plots (Plots 3 & 4)...")
    copy_required_plots()

    print("\n4. Computing feature importance (Plot 5)...")
    feature_names = list(reg_features.X_train.columns)
    plot_feature_importance(
        reg_model,
        feature_names,
        reg_features.X_test,
        reg_bundle.y_test,
        utils.FIGURES_DIR
    )

    print("\n5. Creating comparison tables (Tables 1 & 2)...")
    create_comparison_tables(utils.TABLES_DIR)

    print("\n6. Creating summary checklist...")
    create_summary_document()

    print("\n" + "="*70)
    print("✓ ALL ARTIFACTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review all plots in: reports/figures/")
    print("2. Review all tables in: reports/tables/")
    print("3. Check the checklist: reports/FINAL_REPORT_CHECKLIST.txt")
    print("4. Use these artifacts to write your final report and poster")
    print("="*70)


if __name__ == "__main__":
    main()

