import pandas as pd

# Load and verify the dataset
df = pd.read_csv('heart.csv')

print("=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)
print("File loaded successfully!")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")

# Correcting column name from 'target' to 'HeartDisease' based on dataset preview
target_col = 'HeartDisease' 
if target_col in df.columns:
    print(f"\nTarget distribution ({target_col}):\n{df[target_col].value_counts()}")
else:
    print(f"\nColumn '{target_col}' not found. Available columns: {list(df.columns)}")
