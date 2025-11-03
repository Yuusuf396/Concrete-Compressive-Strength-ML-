import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, mean_squared_error, r2_score

# Load data (project-root relative)
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'concrete_data.csv')
df = pd.read_csv(DATA_PATH)
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_aggregate', 'fine_aggregate', 'age', 'strength'
]

# Create plots directory at repo root if it doesn't exist
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot 1: Target Distribution
plt.figure(figsize=(5,4))
median_strength = df['strength'].median()
df['strength_class'] = (df['strength'] > median_strength).astype(int)
sns.countplot(x='strength_class', data=df, palette='pastel')
plt.title("Target Distribution: High vs Low Strength")
plt.xlabel("Strength Class (0=Low, 1=High)")
plt.ylabel("Count")
plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'))
plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Concrete Features")
plt.tight_layout()  # Added to prevent label cutoff
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
plt.close()

# Plot 3: Confusion Matrix
X = df.drop(['strength', 'strength_class'], axis=1)
y = df['strength_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()

# Print classification report for detailed metrics
report = classification_report(y_test, model.predict(X_test))
with open(os.path.join(PLOTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report for Logistic Regression\n")
    f.write("===========================================\n\n")
    f.write(report)

# Plot 4: Residuals vs Predicted (Regression Analysis)
y_reg = df['strength']
X_reg = df.drop(['strength', 'strength_class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=y_test - y_pred, color='teal', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.title("Residuals vs Predicted – Linear Regression")
plt.xlabel("Predicted Strength (MPa)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()

# Add trend line to check for systematic bias
z = np.polyfit(y_pred, y_test - y_pred, 1)
p = np.poly1d(z)
plt.plot(y_pred, p(y_pred), "r--", alpha=0.8, label='Trend')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'residuals_plot.png'))
plt.close()

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Save regression analysis report
with open(os.path.join(PLOTS_DIR, 'regression_report.txt'), 'w') as f:
    f.write("Regression Analysis Report\n")
    f.write("========================\n\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} MPa\n")
    f.write(f"R-squared Score: {r2:.3f}\n\n")
    f.write("Residuals Analysis:\n")
    f.write(f"Mean Residual: {np.mean(y_test - y_pred):.2f} MPa\n")
    f.write(f"Std of Residuals: {np.std(y_test - y_pred):.2f} MPa\n")
    f.write(f"Max Absolute Residual: {np.max(np.abs(y_test - y_pred)):.2f} MPa\n")
    
    # Check for systematic bias
    if abs(np.mean(y_test - y_pred)) > rmse/5:  # If mean residual is more than 20% of RMSE
        f.write("\nPotential systematic bias detected: Mean residual is significantly different from zero\n")
    if abs(z[0]) > 0.1:  # If trend line has significant slope
        f.write("\nNon-linear pattern detected in residuals: Consider using non-linear models (Decision Trees or Neural Networks)\n")

print("Plots have been saved in the 'plots' directory.")
print("Classification report has been saved as 'plots/classification_report.txt'")
print("Regression analysis report has been saved as 'plots/regression_report.txt'")