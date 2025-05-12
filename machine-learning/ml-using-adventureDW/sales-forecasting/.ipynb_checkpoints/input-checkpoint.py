import pandas as pd
import numpy as np
from nbformat import v4 as nbf
import nbformat
import os

# --- 1. Sales Forecasting Dataset ---
months = pd.date_range(start="2022-01-01", end="2024-12-01", freq='MS')
categories = ['Bikes', 'Accessories', 'Clothing']
sales_data = []

np.random.seed(42)
for category in categories:
    base = np.random.randint(20000, 40000)
    for month in months:
        season = 1.3 if month.month in [5,6,7] else 0.9 if month.month in [1,2] else 1.0
        trend = 1 + 0.05 * (month.year - 2022)
        value = base * season * trend * np.random.normal(1.0, 0.1)
        sales_data.append({
            "Month": month.strftime("%Y-%m"),
            "Category": category,
            "SalesAmount": round(value, 2)
        })

df_sales = pd.DataFrame(sales_data)
df_sales.to_csv("sales_forecasting.csv", index=False)

# --- 2. Market Basket Dataset ---
products = ["Helmet", "Bottle", "Jersey", "Bike", "Gloves", "Shoes"]
transactions = []

np.random.seed(0)
for order_id in range(1, 1001):
    n_items = np.random.randint(1, 5)
    chosen = list(np.random.choice(products, n_items, replace=False))
    for item in chosen:
        transactions.append({"OrderID": order_id, "Product": item})

df_basket = pd.DataFrame(transactions)
df_basket.to_csv("market_basket.csv", index=False)

# --- 3. Sales Forecasting Notebook ---
nb_sales = nbf.new_notebook()
nb_sales.cells = [
    nbf.new_markdown_cell("# 📈 Sales Forecasting with Linear Regression and Prophet"),
    nbf.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="whitegrid")"""),
    nbf.new_code_cell("""df = pd.read_csv("sales_forecasting.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.head()"""),
    nbf.new_code_cell("""# Filter for one category
category = 'Bikes'
df_cat = df[df['Category'] == category].copy()
df_cat['MonthIndex'] = np.arange(len(df_cat))"""),
    nbf.new_code_cell("""# Linear Regression
model = LinearRegression()
model.fit(df_cat[['MonthIndex']], df_cat['SalesAmount'])
df_cat['Forecast'] = model.predict(df_cat[['MonthIndex']])

plt.figure(figsize=(10,5))
plt.plot(df_cat['Month'], df_cat['SalesAmount'], label='Actual')
plt.plot(df_cat['Month'], df_cat['Forecast'], label='Forecast')
plt.legend()
plt.title(f"{category} Sales Forecast")
plt.show()"""),
    nbf.new_code_cell("""# Optional: Prophet forecasting
try:
    from prophet import Prophet
    df_prophet = df_cat[['Month', 'SalesAmount']].rename(columns={"Month": "ds", "SalesAmount": "y"})
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    fig = m.plot(forecast)
except ImportError:
    print("Prophet not installed. Skipping Prophet forecast.")""")
]

with open("sales_forecasting.ipynb", "w") as f:
    nbformat.write(nb_sales, f)

# --- 4. Market Basket Analysis Notebook ---
nb_basket = nbf.new_notebook()
nb_basket.cells = [
    nbf.new_markdown_cell("# 🛒 Market Basket Analysis using Apriori"),
    nbf.new_code_cell("""import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder"""),
    nbf.new_code_cell("""df = pd.read_csv("market_basket.csv")
df.head()"""),
    nbf.new_code_cell("""# Group into transactions
transactions = df.groupby("OrderID")['Product'].apply(list).tolist()"""),
    nbf.new_code_cell("""# Encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)"""),
    nbf.new_code_cell("""# Apriori
frequent = apriori(df_encoded, min_support=0.02, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head()""")
]

with open("market_basket_analysis.ipynb", "w") as f:
    nbformat.write(nb_basket, f)

print("✅ Files generated: sales_forecasting.csv, market_basket.csv, and two Jupyter notebooks.")
