import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("concrete_data.csv")
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_aggregate', 'fine_aggregate', 'age', 'strength'
]
median_strength = df['strength'].median()
df['strength_class'] = (df['strength'] > median_strength).astype(int)

# Plot
plt.figure(figsize=(5,4))
sns.countplot(x='strength_class', data=df, palette='pastel')
plt.title("Target Distribution: High vs Low Strength")
plt.xlabel("Strength Class (0=Low, 1=High)")
plt.ylabel("Count")
plt.savefig('plots/target_distribution.png')
plt.close()
