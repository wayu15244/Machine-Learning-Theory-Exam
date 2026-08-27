import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Slide Example: Points A to F with K=2 (Slide 8-11)
# -------------------------------------------------------------
data_points = {
    'A': np.array([2, 3]),
    'B': np.array([3, 4]),
    'C': np.array([4, 5]),
    'D': np.array([8, 8]),
    'E': np.array([9, 9]),
    'F': np.array([8, 10])
}

c1_init = np.array([2.0, 3.0])
c2_init = np.array([9.0, 9.0])

print("=== K-Means Clustering: Slide Example (Slide 8-11) ===")
print("Data Points:")
for p, coord in data_points.items():
    print(f"  Point {p}: {coord}")

print(f"\nInitial Centroids: C1 = {c1_init}, C2 = {c2_init}")

# Iteration 1
print("\n--- Iteration 1: Distance Calculation ---")
print(f"{'Point':<6} {'Coord':<10} {'Dist to C1':<14} {'Dist to C2':<14} {'Assigned Cluster':<18}")
print("-" * 62)

cluster1_pts = []
cluster2_pts = []

for p, coord in data_points.items():
    d1 = np.sqrt(np.sum((coord - c1_init)**2))
    d2 = np.sqrt(np.sum((coord - c2_init)**2))
    assigned = 'Cluster 1 (C1)' if d1 < d2 else 'Cluster 2 (C2)'
    if d1 < d2:
        cluster1_pts.append(coord)
    else:
        cluster2_pts.append(coord)
    print(f"{p:<6} ({coord[0]},{coord[1]}){'':<4} {d1:.2f}{'':<8} {d2:.2f}{'':<8} {assigned:<18}")

# Update Centroids
c1_new = np.mean(cluster1_pts, axis=0)
c2_new = np.mean(cluster2_pts, axis=0)

print("\n--- Centroid Update ---")
print(f"Cluster 1: {[list(pt) for pt in cluster1_pts]} -> C1 new = ({c1_new[0]:.2f}, {c1_new[1]:.2f})")
print(f"Cluster 2: {[list(pt) for pt in cluster2_pts]} -> C2 new = ({c2_new[0]:.2f}, {c2_new[1]:.2f})")

# -------------------------------------------------------------
# 2. Plot: Step-by-Step K-Means Visualization
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

pts_array = np.array(list(data_points.values()))
pt_labels = list(data_points.keys())

# Subplot 1: Initial State & Iteration 1 Assignment
ax1.scatter([c[0] for c in cluster1_pts], [c[1] for c in cluster1_pts], color='#1d3557', s=160, label='Assigned Cluster 1', zorder=4)
ax1.scatter([c[0] for c in cluster2_pts], [c[1] for c in cluster2_pts], color='#2a9d8f', s=160, label='Assigned Cluster 2', zorder=4)

for p, coord in data_points.items():
    ax1.annotate(f"{p} ({coord[0]},{coord[1]})", (coord[0] + 0.2, coord[1] - 0.1), fontsize=10.5, fontweight='bold')

# Initial Centroids
ax1.scatter([c1_init[0]], [c1_init[1]], color='#e63946', marker='X', s=280, edgecolors='black', linewidth=1.5,
            label=f'Initial C1 (2,3)', zorder=5)
ax1.scatter([c2_init[0]], [c2_init[1]], color='#9d0208', marker='X', s=280, edgecolors='black', linewidth=1.5,
            label=f'Initial C2 (9,9)', zorder=5)

ax1.set_xlim(0, 11)
ax1.set_ylim(1, 11)
ax1.set_title('K-Means Iteration 1: Initial Centroids & Assignment\nCluster 1: {A, B, C} | Cluster 2: {D, E, F}', fontsize=12, fontweight='bold')
ax1.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Centroid Update & Convergence
ax2.scatter([c[0] for c in cluster1_pts], [c[1] for c in cluster1_pts], color='#1d3557', s=160, label='Cluster 1 Points', zorder=4)
ax2.scatter([c[0] for c in cluster2_pts], [c[1] for c in cluster2_pts], color='#2a9d8f', s=160, label='Cluster 2 Points', zorder=4)

for p, coord in data_points.items():
    ax2.annotate(f"{p} ({coord[0]},{coord[1]})", (coord[0] + 0.2, coord[1] - 0.1), fontsize=10.5, fontweight='bold')

# Draw motion arrow of Centroids
ax2.scatter([c1_init[0]], [c1_init[1]], color='gray', marker='X', s=160, alpha=0.5, label='Old Centroids')
ax2.scatter([c2_init[0]], [c2_init[1]], color='gray', marker='X', s=160, alpha=0.5)

ax2.scatter([c1_new[0]], [c1_new[1]], color='#e63946', marker='X', s=300, edgecolors='black', linewidth=2,
            label=f'Updated C1 ({c1_new[0]:.2f}, {c1_new[1]:.2f})', zorder=6)
ax2.scatter([c2_new[0]], [c2_new[1]], color='#9d0208', marker='X', s=300, edgecolors='black', linewidth=2,
            label=f'Updated C2 ({c2_new[0]:.2f}, {c2_new[1]:.2f})', zorder=6)

# Arrows
ax2.annotate('', xy=c1_new, xytext=c1_init, arrowprops=dict(facecolor='#e63946', edgecolor='#e63946', width=2, headwidth=8))
ax2.annotate('', xy=c2_new, xytext=c2_init, arrowprops=dict(facecolor='#9d0208', edgecolor='#9d0208', width=2, headwidth=8))

# Dashed lines connecting points to new centroids
for pt in cluster1_pts:
    ax2.plot([pt[0], c1_new[0]], [pt[1], c1_new[1]], color='#1d3557', linestyle=':', alpha=0.6)
for pt in cluster2_pts:
    ax2.plot([pt[0], c2_new[0]], [pt[1], c2_new[1]], color='#2a9d8f', linestyle=':', alpha=0.6)

ax2.set_xlim(0, 11)
ax2.set_ylim(1, 11)
ax2.set_title('K-Means Centroid Update & Final Convergence\nC1: (3.00, 4.00) & C2: (8.33, 9.00)', fontsize=12, fontweight='bold')
ax2.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/kmeans_clustering_slide.png', dpi=300)
plt.close()
print("Saved figure to figures/kmeans_clustering_slide.png")

# -------------------------------------------------------------
# 3. Good vs Poor Clustering Figure (Slide 12)
# -------------------------------------------------------------
np.random.seed(42)
fig, (ax_g, ax_p) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Generate 3 well-separated blobs
b1 = np.random.randn(30, 2) * 0.4 + [2, 2]
b2 = np.random.randn(30, 2) * 0.5 + [5, 6]
b3 = np.random.randn(30, 2) * 0.4 + [8, 8]

# Good Clustering
ax_g.scatter(b1[:, 0], b1[:, 1], c='#e63946', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 1')
ax_g.scatter(b2[:, 0], b2[:, 1], c='#1d3557', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 2')
ax_g.scatter(b3[:, 0], b3[:, 1], c='#2a9d8f', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 3')

# Centroids
ax_g.scatter([np.mean(b1[:, 0])], [np.mean(b1[:, 1])], c='#9d0208', marker='X', s=250, edgecolors='black', linewidth=1.5)
ax_g.scatter([np.mean(b2[:, 0])], [np.mean(b2[:, 1])], c='#03045e', marker='X', s=250, edgecolors='black', linewidth=1.5)
ax_g.scatter([np.mean(b3[:, 0])], [np.mean(b3[:, 1])], c='#1b4332', marker='X', s=250, edgecolors='black', linewidth=1.5)

ax_g.set_title('Good Clustering (Well-separated, tight clusters)',
               fontsize=12, fontweight='bold', color='#1b4332')
ax_g.set_xlim(0, 10)
ax_g.set_ylim(0, 10)
ax_g.legend(loc='upper left')
ax_g.grid(True, linestyle='--', alpha=0.6)

# Poor Clustering (Bad initialization or ill-suited K)
all_pts = np.vstack([b1, b2, b3])
bad_colors = ['#e63946', '#1d3557', '#2a9d8f']
np.random.shuffle(all_pts)
p_c1, p_c2, p_c3 = all_pts[:30], all_pts[30:60], all_pts[60:]

ax_p.scatter(p_c1[:, 0], p_c1[:, 1], c='#e63946', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 1')
ax_p.scatter(p_c2[:, 0], p_c2[:, 1], c='#1d3557', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 2')
ax_p.scatter(p_c3[:, 0], p_c3[:, 1], c='#2a9d8f', s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='Cluster 3')

ax_p.scatter([3.5], [6.5], c='#9d0208', marker='X', s=250, edgecolors='black', linewidth=1.5)
ax_p.scatter([5.0], [5.0], c='#03045e', marker='X', s=250, edgecolors='black', linewidth=1.5)
ax_p.scatter([7.0], [6.5], c='#1b4332', marker='X', s=250, edgecolors='black', linewidth=1.5)

ax_p.set_title('Poor Clustering (Overlapping clusters, improper K/centroids)',
               fontsize=12, fontweight='bold', color='#b7094c')
ax_p.set_xlim(0, 10)
ax_p.set_ylim(0, 10)
ax_p.legend(loc='upper left')
ax_p.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/kmeans_good_vs_poor.png', dpi=300)
plt.close()
print("Saved figure to figures/kmeans_good_vs_poor.png")
