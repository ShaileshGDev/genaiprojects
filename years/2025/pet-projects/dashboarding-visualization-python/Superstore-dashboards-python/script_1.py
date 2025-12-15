
import pandas as pd
import numpy as np

# Try with latin-1 encoding
df = pd.read_csv('Sample-Superstore.csv', encoding='latin-1')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head(3))
print("\nColumn Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())
print("\nUnique Values per Column:")
print(df.nunique())
