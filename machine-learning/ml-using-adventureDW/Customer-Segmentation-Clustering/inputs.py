import pandas as pd
import numpy as np
from nbformat import v4 as nbf
import nbformat

# Step 1: Generate synthetic customer segmentation data
np.random.seed(42)
num_customers = 1000
customer_ids = np.arange(1, num_customers + 1)
total_spend = np.random.gamma(shape=2.0, scale=500.0, size=num_customers).round(2)
num_orders = np.random.poisson(lam=5, size=num_customers)
avg_order_value = (total_spend / np.maximum(num_orders, 1)).round(2)
recency_days = np.random.randint(1, 365, size=num_customers)
regions = np.random.choice(['North America', 'Europe', 'Pacific', 'South America'], size=num_customers)

df_customers = pd.DataFrame({
    'CustomerID': customer_ids,
    'TotalSpend': total_spend,
    'NumOrders': num_orders,
    'AvgOrderValue': avg_order_value,
    'RecencyDays': recency_days,
    'Region': regions
})

df_customers.to_csv("customer_segmentation.csv", index=False)

# Step 2: Create Jupyter Notebook
nb = nbf.new_notebook()
cells = []

cells.append(nbf.new_markdown_cell("# Customer Segmentation using K-Means Clustering\nThis notebook demonstrates how to apply clustering on customer purchase behavior using synthetic data based on the AdventureWorks data warehouse."))

cells.append(nbf.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set(style="whitegrid")"""))

cells.append(nbf.new_code_cell("""# Load dataset
df = pd.read_csv("customer_segmentation.csv")
df.head()"""))

cells.append(nbf.new_code_cell("""# Encode categorical data
df_encoded = pd.get_dummies(df, columns=['Region'], drop_first=True)

# Standardize numerical features
features = ['TotalSpend', 'NumOrders', 'AvgOrderValue', 'RecencyDays'] + [col for col in df_encoded.columns if col.startswith('Region_')]
X = df_encoded[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)"""))

cells.append(nbf.new_code_cell("""# Elbow method to find optimal k
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()"""))

cells.append(nbf.new_code_cell("""# Fit KMeans with k=4 (example)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='tab10')
plt.title('Customer Segments (PCA Projection)')
plt.show()"""))

cells.append(nbf.new_code_cell("""# Analyze cluster characteristics
df.groupby('Cluster')[['TotalSpend', 'NumOrders', 'AvgOrderValue', 'RecencyDays']].mean().round(2)"""))

nb.cells = cells

with open("customer_segmentation_kmeans.ipynb", "w") as f:
    nbformat.write(nb, f)

print("✅ Files generated: 'customer_segmentation.csv' and 'customer_segmentation_kmeans.ipynb'")
