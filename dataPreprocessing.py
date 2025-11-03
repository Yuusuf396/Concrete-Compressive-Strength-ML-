import pandas as pd
df = pd.read_csv("concrete_data.csv")
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_aggregate', 'fine_aggregate', 'age', 'strength'
]
median_strength = df['strength'].median()
df['strength_class'] = (df['strength'] > median_strength).astype(int)
df.info(); df.describe(); df['strength_class'].value_counts(normalize=True)
