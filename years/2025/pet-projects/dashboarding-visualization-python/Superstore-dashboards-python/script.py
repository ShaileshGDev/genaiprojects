
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('Sample-Superstore.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn Names and Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nUnique Values per Column:")
print(df.nunique())
