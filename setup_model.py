import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('Mall_Customers.csv')

# Use relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Retrain model with 5 clusters (optimal based on Elbow method in notebook)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save model and scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.joblib.pkl')

print("Model retrained with 5 clusters and saved successfully.")
print("Cluster Centers (Scaled):")
print(kmeans.cluster_centers_)
print("Cluster Centers (Original):")
print(scaler.inverse_transform(kmeans.cluster_centers_))
