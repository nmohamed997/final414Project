import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Online Retail
online_retail_path = 'Online_retail.csv'
df = pd.read_csv(online_retail_path, encoding='UTF-8', low_memory=False)

# Amazon Sale Report
amazon_data_path = 'Amazon_Sale_Report.csv'
amazon_data = pd.read_csv(amazon_data_path, low_memory=False)

# Online Retail Dataset
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Amazon Dataset
amazon_data = amazon_data.dropna(subset=['Amount'])
amazon_data = amazon_data[amazon_data['Qty'] > 0]
amazon_data['TotalPrice'] = amazon_data['Qty'] * amazon_data['Amount']

# Online Retail - Customer Level Aggregation
retail_customers = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum',
    'Quantity': 'sum'
}).reset_index()
retail_customers.rename(columns={'InvoiceNo': 'purchase_frequency'}, inplace=True)

# Amazon - Customer Level Aggregation
amazon_customers = amazon_data.groupby('Order ID').agg({
    'Qty': 'sum',
    'TotalPrice': 'sum'
}).reset_index()
amazon_customers.rename(columns={'Qty': 'total_quantity'}, inplace=True)


# Online Retail: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(retail_customers.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Online Retail)')
plt.show()

# Amazon: Distribution of Total Spending
plt.figure(figsize=(10, 6))
sns.histplot(amazon_customers['TotalPrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Total Spending (Amazon)')
plt.xlabel('Total Spending')
plt.ylabel('Frequency')
plt.show()

# Online Retail Clustering
scaler = StandardScaler()
retail_features = ['purchase_frequency', 'TotalPrice', 'Quantity']
retail_customers[retail_features] = scaler.fit_transform(retail_customers[retail_features])

# Determine optimal number of clusters for Online Retail
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(retail_customers[retail_features])
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Online Retail Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k_retail = 4
kmeans_retail = KMeans(n_clusters=optimal_k_retail, random_state=42)
retail_customers['cluster'] = kmeans_retail.fit_predict(retail_customers[retail_features])

# Visualize Clusters for Online Retail
plt.figure(figsize=(10, 6))
sns.scatterplot(data=retail_customers, x='purchase_frequency', y='TotalPrice', hue='cluster', palette='viridis')
plt.title('Customer Segments (Online Retail)')
plt.xlabel('Purchase Frequency')
plt.ylabel('Total Spending')
plt.legend(title='Cluster')
plt.show()

# Amazon Clustering
amazon_features = ['total_quantity', 'TotalPrice']
amazon_customers[amazon_features] = scaler.fit_transform(amazon_customers[amazon_features])

# Determine optimal number of clusters for Amazon
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(amazon_customers[amazon_features])
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Amazon Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k_amazon = 4
kmeans_amazon = KMeans(n_clusters=optimal_k_amazon, random_state=42)
amazon_customers['cluster'] = kmeans_amazon.fit_predict(amazon_customers[amazon_features])

# Visualize Clusters for Amazon
plt.figure(figsize=(10, 6))
sns.scatterplot(data=amazon_customers, x='total_quantity', y='TotalPrice', hue='cluster', palette='viridis')
plt.title('Customer Segments (Amazon)')
plt.xlabel('Total Quantity')
plt.ylabel('Total Spending')
plt.legend(title='Cluster')
plt.show()

# Regression & Classification (Online Retail Example)
X_reg = retail_customers[['purchase_frequency', 'Quantity']]
y_reg = retail_customers['TotalPrice']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)

print("Online Retail Regression Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_reg)}")
print(f"R-squared: {r2_score(y_test_reg, y_pred_reg)}")

# Classification
retail_customers['spender_type'] = (retail_customers['TotalPrice'] > retail_customers['TotalPrice'].median()).astype(int)
X_class = retail_customers[['purchase_frequency', 'Quantity']]
y_class = retail_customers['spender_type']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_class, y_train_class)
y_pred_class = classifier.predict(X_test_class)

print("\nOnline Retail Classification Results:")
print(classification_report(y_test_class, y_pred_class))


print("\nRecommendations:")
print("- Use clusters to identify and target high-spending customers with promotions.")
print("- Leverage purchasing frequency to predict future spending and loyalty.")
print("- Prioritize product categories and regions based on spending patterns.")
