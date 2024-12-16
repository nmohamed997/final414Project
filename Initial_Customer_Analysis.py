import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Online_retail.csv', encoding='UTF-8')

# Data Cleaning
# Drop rows with missing CustomerID and filter out negative quantities
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Feature Engineering
# Calculate total purchase value per item
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Aggregate data at the customer level
customer_data = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',           # Purchase frequency (number of invoices)
    'TotalPrice': 'sum',              # Total expenditure
    'StockCode': lambda x: x.mode()[0] # Most frequently purchased item
}).reset_index()

# Rename columns for clarity
customer_data.columns = ['CustomerID', 'purchase_frequency', 'total_expenditure', 'preferred_item']

# One-hot encode 'preferred_item' (the most frequently purchased item)
customer_data = pd.get_dummies(customer_data, columns=['preferred_item'], drop_first=True)

# Scaling numeric data
scaler = StandardScaler()
numeric_features = ['purchase_frequency', 'total_expenditure']
customer_data[numeric_features] = scaler.fit_transform(customer_data[numeric_features])

# Determine Optimal Number of Clusters (Elbow Method)
inertia_values = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data[numeric_features])
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Select optimal k based on the elbow plot
optimal_k = 4  # Adjust this after reviewing the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data[numeric_features])

# Display cluster summary
cluster_summary = customer_data.groupby('cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

# Display sample data for each cluster
for cluster in range(optimal_k):
    print(f"\nSample data for Cluster {cluster}")
    print(customer_data[customer_data['cluster'] == cluster].head(5))

# Visualize Clusters
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['purchase_frequency'], customer_data['total_expenditure'], c=customer_data['cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Purchase Frequency (scaled)')
plt.ylabel('Total Expenditure (scaled)')
plt.colorbar(label='Cluster')
plt.show()
