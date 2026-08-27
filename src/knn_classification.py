import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Slide Example 1: Points A to E with Query Points P1, P2
# -------------------------------------------------------------
points = {
    'A': (2, 3, 'Class A'),
    'B': (3, 5, 'Class B'),
    'C': (6, 8, 'Class B'),
    'D': (7, 6, 'Class B'),
    'E': (5, 4, 'Class A')
}

queries = {
    'P1': (4, 5),
    'P2': (6, 5)
}

print("=== KNN Classification: Example 1 (Points A-E) ===")
for q_name, (qx, qy) in queries.items():
    print(f"\nQuery Point {q_name} = ({qx}, {qy}):")
    distances = []
    for p_name, (px, py, cls) in points.items():
        d = np.sqrt((qx - px)**2 + (qy - py)**2)
        distances.append((p_name, px, py, cls, d))
    
    # Sort by distance
    distances.sort(key=lambda item: item[4])
    print(f"{'Point':<6} {'Coord':<10} {'Class':<10} {'Distance':<10}")
    print("-" * 38)
    for p_name, px, py, cls, d in distances:
        print(f"{p_name:<6} ({px}, {py}){'':<4} {cls:<10} {d:.4f}")
    
    # K=1
    print(f"-> Prediction (K=1): {distances[0][3]}")
    
    # K=3
    top3_classes = [item[3] for item in distances[:3]]
    k3_pred = max(set(top3_classes), key=top3_classes.count)
    print(f"-> Top 3: {[(item[0], item[3], round(item[4], 3)) for item in distances[:3]]}")
    print(f"-> Prediction (K=3): {k3_pred}")

# -------------------------------------------------------------
# 2. Slide Example 2: Weighted KNN Classification (Slide 24-26)
# -------------------------------------------------------------
train_data_weighted = [
    (1, 1, 2, 0), (2, 2, 3, 0), (3, 1, 1, 0),
    (4, 6, 5, 1), (5, 7, 6, 1), (6, 8, 5, 1), (7, 5, 6, 1),
    (8, 3, 9, 2), (9, 4, 8, 2), (10, 3, 7, 2), (11, 2, 2, 0),
    (12, 9, 4, 1), (13, 5, 9, 2), (14, 4, 7, 2), (15, 6, 7, 1)
]

query_weighted = np.array([4, 6])
epsilon = 1e-5
p = 2
k = 5

print("\n=== Weighted KNN Classification: Example 2 (Slide 24-26) ===")
dist_weights = []
for idx, x1, x2, cls in train_data_weighted:
    d = np.sqrt((query_weighted[0] - x1)**2 + (query_weighted[1] - x2)**2)
    w = 1.0 / (d**p + epsilon)
    dist_weights.append((idx, x1, x2, cls, d, w))

dist_weights.sort(key=lambda item: item[4])
print(f"Query x = {query_weighted}, K = {k}")
print(f"{'Rank':<5} {'Index':<6} {'Coord':<10} {'Class':<8} {'Distance':<10} {'Weight':<10}")
print("-" * 52)
for rank, item in enumerate(dist_weights[:k], 1):
    print(f"{rank:<5} {item[0]:<6} ({item[1]},{item[2]}){'':<4} {item[3]:<8} {item[4]:.4f}{'':<4} {item[5]:.4f}")

# Aggregate weights per class
class_scores = {0: 0.0, 1: 0.0, 2: 0.0}
for item in dist_weights[:k]:
    class_scores[item[3]] += item[5]

print("\nClass Scores (Sum of Weights):")
for cls, score in class_scores.items():
    print(f"Class {cls}: {score:.4f}")
pred_class = max(class_scores, key=class_scores.get)
print(f"Predicted Class: {pred_class} with score {class_scores[pred_class]:.4f}")

# -------------------------------------------------------------
# 3. Generate Plot Figure for Report
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Plot training points
class_a_x = [points[k][0] for k in points if points[k][2] == 'Class A']
class_a_y = [points[k][1] for k in points if points[k][2] == 'Class A']
class_b_x = [points[k][0] for k in points if points[k][2] == 'Class B']
class_b_y = [points[k][1] for k in points if points[k][2] == 'Class B']

ax.scatter(class_a_x, class_a_y, color='#e63946', s=150, edgecolors='black', linewidth=1.5, label='Class A (Training)', zorder=4)
ax.scatter(class_b_x, class_b_y, color='#1d3557', s=150, edgecolors='black', linewidth=1.5, label='Class B (Training)', zorder=4)

# Labels on training points
for k_name, (px, py, cls) in points.items():
    ax.annotate(f"{k_name} ({px},{py})", (px + 0.15, py + 0.15), fontsize=11, fontweight='bold',
                color='#e63946' if cls == 'Class A' else '#1d3557')

# Plot Query Points P1 and P2
ax.scatter([queries['P1'][0]], [queries['P1'][1]], color='#2a9d8f', marker='*', s=350, edgecolors='black',
           linewidth=1.5, label='Query P1 (4, 5)', zorder=5)
ax.scatter([queries['P2'][0]], [queries['P2'][1]], color='#e76f51', marker='^', s=250, edgecolors='black',
           linewidth=1.5, label='Query P2 (6, 5)', zorder=5)

# Circle around P1 for K=3
# Radius for K=3 at P1 is distance to A (2.828)
circle_p1 = plt.Circle(queries['P1'], 2.85, color='#2a9d8f', fill=False, linestyle='--', linewidth=2, label='P1 K=3 Radius (d≈2.83)')
ax.add_patch(circle_p1)

# Draw connecting lines from P1 to its 3 nearest neighbors (B, E, A)
for target in ['B', 'E', 'A']:
    tx, ty, _ = points[target]
    ax.plot([queries['P1'][0], tx], [queries['P1'][1], ty], color='#2a9d8f', linestyle=':', linewidth=1.8, alpha=0.8)

ax.set_xlim(0, 9)
ax.set_ylim(0, 10)
ax.set_title('K-Nearest Neighbor (KNN) Classification\nQuery Point P1 (4,5) with K=3 -> Predicted: Class A', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Feature X', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature Y', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/knn_classification_slide.png', dpi=300)
plt.close()
print("Saved figure to figures/knn_classification_slide.png")
