import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Mall Customers Segmentation using K-Means (K=5)
# Demonstrating 05_K-Means.ipynb
# -------------------------------------------------------------
np.random.seed(42)

# Generate synthetic Mall Customer clusters corresponding to 5 segments:
# 1. Low Income, Low Spending (Sensible)
c1 = np.column_stack([np.random.normal(25, 6, 40), np.random.normal(20, 7, 40)])
# 2. Low Income, High Spending (Careless)
c2 = np.column_stack([np.random.normal(25, 6, 40), np.random.normal(80, 8, 40)])
# 3. Middle Income, Middle Spending (Standard)
c3 = np.column_stack([np.random.normal(55, 7, 40), np.random.normal(50, 7, 40)])
# 4. High Income, Low Spending (Careful)
c4 = np.column_stack([np.random.normal(85, 8, 40), np.random.normal(18, 6, 40)])
# 5. High Income, High Spending (Target / VIP)
c5 = np.column_stack([np.random.normal(86, 9, 40), np.random.normal(82, 7, 40)])

X_mall = np.vstack([c1, c2, c3, c4, c5])

def kmeans_scratch(X, k=5, max_iter=100, tol=1e-4):
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices].copy()
    
    for it in range(max_iter):
        # Distance calculation
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else centroids[j] for j in range(k)])
        shift = np.sqrt(((new_centroids - centroids)**2).sum())
        centroids = new_centroids
        if shift < tol:
            break
    return labels, centroids

labels, centroids = kmeans_scratch(X_mall, k=5)

# Test instances from Colab: CustomerID 201-205
# 201: Female, 24, Income 38, Spending 18 -> Low spending, Low-mid income
# 202: Female, 31, Income 38, Spending 75 -> High spending, Low-mid income
# 203: Female, 45, Income 55, Spending 52 -> Standard
# 204: Male, 52, Income 72, Spending 30 -> Careful
# 205: Female, 28, Income 90, Spending 92 -> VIP / Target
test_customers = np.array([
    [38, 18],
    [38, 75],
    [55, 52],
    [72, 30],
    [90, 92]
])
test_ids = [201, 202, 203, 204, 205]

test_dists = np.sqrt(((test_customers[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
test_labels = np.argmin(test_dists, axis=1)

print("K-Means Customer Segmentation Finished.")
print("Centroids:")
for idx, cent in enumerate(centroids):
    print(f"  Cluster {idx+1}: Income=${cent[0]:.1f}k, Spending={cent[1]:.1f}")

# Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.figure(figsize=(10, 7), dpi=300)

cluster_colors = ['#e63946', '#457b9d', '#2a9d8f', '#e76f51', '#9b5de5']
cluster_names = [f'Cluster {i+1}' for i in range(5)]

for i in range(5):
    pts = X_mall[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], c=cluster_colors[i], s=55, alpha=0.75, label=f'Cluster {i+1}', edgecolors='white', linewidth=0.5)

# Centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=260, edgecolors='yellow', linewidth=2, label='Centroids', zorder=5)

# Test Points
plt.scatter(test_customers[:, 0], test_customers[:, 1], c='gold', marker='*', s=300, edgecolors='black', linewidth=1.5, label='Test Customers (201-205)', zorder=6)
for i, (cid, pt) in enumerate(zip(test_ids, test_customers)):
    plt.annotate(f"ID {cid}", (pt[0] + 1.5, pt[1] - 1), fontsize=10, fontweight='bold', color='black')

plt.title('Customer Segmentation using K-Means (K=5)\nAnnual Income vs Spending Score (1-100)', fontsize=13, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=11, fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontsize=11, fontweight='bold')
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/mall_customers_kmeans.png', dpi=300)
plt.close()
print("Saved figure to figures/mall_customers_kmeans.png")
