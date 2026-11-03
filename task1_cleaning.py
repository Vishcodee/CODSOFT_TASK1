import numpy as np
import pandas as pd

# 1. Create and Save Raw Dataset
np.random.seed(42)
n_rows = 1200

data = {
    'InvoiceNo': [f'INV{1000 + i}' if i % 15 != 0 else np.nan for i in range(n_rows)],
    'CustomerID': [f'CUST_{np.random.randint(100, 250)}' if i % 10 != 0 else np.nan for i in range(n_rows)],
    'Age': [np.random.choice([np.random.randint(18, 70), np.nan, -5, 150]) for _ in range(n_rows)],
    'Gender': np.random.choice(['Male', 'Female', 'unknown', None], size=n_rows),
    'PurchaseDate': pd.date_range(start='2025-01-01', periods=n_rows, freq='h').astype(str),
    'AmountSpent': [round(np.random.uniform(10.0, 500.0), 2) if i % 20 != 0 else -99.0 for i in range(n_rows)],
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'invalid_cat'], size=n_rows)
}

df_raw = pd.DataFrame(data)
df_raw = pd.concat([df_raw, df_raw.iloc[:35]], ignore_index=True)
df_raw.to_csv('raw_retail_data.csv', index=False)
print("--- 1. Raw Dataset Loaded ---")
print(f"Total initial rows: {len(df_raw)}")
print(f"Missing values per column:\n{df_raw.isnull().sum()}")

# 2. Data Cleaning Process
print("\n--- 2. Cleaning Pipeline in Progress ---")
df = pd.read_csv('raw_retail_data.csv')

# Remove duplicate records
init_len = len(df)
df.drop_duplicates(inplace=True)
print(f"[Done] Dropped {init_len - len(df)} duplicate rows.")

# Remove records missing crucial identifiers
df.dropna(subset=['CustomerID', 'InvoiceNo'], inplace=True)
print(f"[Done] Filtered null CustomerID and InvoiceNo.")

# Correct datetime types
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# Filter unrealistic age inputs and impute remaining nulls
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# Filter out negative or zero spend values
df = df[df['AmountSpent'] > 0]

# Standardize categorical anomalies
df['Gender'] = df['Gender'].replace({'unknown': np.nan}).fillna('Not Specified')
df = df[df['Category'] != 'invalid_cat']

# 3. Export Cleaned Data
df.to_csv('cleaned_retail_data.csv', index=False)
print("\n--- 3. Dataset Successfully Cleaned ---")
print(f"Final clean records: {len(df)}")
print(df.head(3))
