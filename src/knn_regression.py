import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Slide Example 1: 5-Features Dataset (Slide 17-19)
# -------------------------------------------------------------
train_data_5f = [
    (1, [2, 3, 4, 2, 3], 50),
    (2, [3, 2, 5, 3, 2], 55),
    (3, [2, 4, 4, 3, 3], 52),
    (4, [3, 3, 5, 2, 4], 58),
    (5, [2, 2, 4, 3, 3], 51),
    (15, [10, 12, 10, 11, 12], 98)
]
test_instance_5f = np.array([3, 3, 4, 3, 3])
k_5f = 3

print("=== KNN Regression Example 1: 5 Features (Slide 17-19) ===")
results_5f = []
for idx, feats, target in train_data_5f:
    d = np.sqrt(np.sum((test_instance_5f - np.array(feats))**2))
    results_5f.append((idx, feats, target, d))

results_5f.sort(key=lambda x: x[3])
print(f"Test Instance = {test_instance_5f}, k = {k_5f}")
for item in results_5f:
    print(f"ID {item[0]:<2}: feats={item[1]}, target={item[2]}, distance={item[3]:.4f}")

top3 = results_5f[:k_5f]
y_hat_simple = np.mean([item[2] for item in top3])
print(f"Selected top {k_5f} neighbors: {[item[0] for item in top3]}")
print(f"Predicted Target y_hat = {y_hat_simple:.2f}")

# -------------------------------------------------------------
# 2. Slide Example 2: Weighted KNN Regression - House Price (Slide 27-29)
# -------------------------------------------------------------
house_data = [
    (1, 50, 1, 950),
    (2, 60, 1, 1050),
    (3, 80, 2, 1300),
    (4, 100, 3, 1600),
    (5, 55, 1, 1000),
    (6, 85, 2, 1350),
    (7, 70, 2, 1200),
    (8, 90, 3, 1500),
    (9, 65, 2, 1150),
    (10, 75, 2, 1250)
]
query_house = np.array([72, 2])
k_house = 5
epsilon = 1e-5

print("\n=== Weighted KNN Regression: House Price (Slide 27-29) ===")
house_res = []
for idx, area, beds, price in house_data:
    d = np.sqrt((query_house[0] - area)**2 + (query_house[1] - beds)**2)
    w = 1.0 / (d**2 + epsilon)
    house_res.append((idx, area, beds, price, d, w))

house_res.sort(key=lambda x: x[4])
top5_house = house_res[:k_house]

num = sum(item[5] * item[3] for item in top5_house)
den = sum(item[5] for item in top5_house)
y_hat_house = num / den

print(f"Query House = {query_house} (72 sqm, 2 beds), k = {k_house}")
print(f"{'Rank':<5} {'Index':<6} {'Area':<6} {'Beds':<6} {'Price':<8} {'Distance':<10} {'Weight':<10}")
print("-" * 55)
for r, item in enumerate(top5_house, 1):
    print(f"{r:<5} {item[0]:<6} {item[1]:<6} {item[2]:<6} {item[3]:<8} {item[4]:.4f}{'':<4} {item[5]:.4f}")

print(f"Numerator: {num:.2f}")
print(f"Denominator: {den:.4f}")
print(f"Predicted Price (y_hat): {y_hat_house:.2f} thousand THB (Slide gives 1217.59)")

# -------------------------------------------------------------
# 3. Generate Plot Figure for KNN Regression
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Subplot 1: House Price vs Area (Slide 27-29)
areas = [h[1] for h in house_data]
prices = [h[3] for h in house_data]
beds = [h[2] for h in house_data]

ax1.scatter(areas, prices, c='#1d3557', s=140, edgecolors='black', label='Training Houses', zorder=4)
for h in house_data:
    ax1.annotate(f"ID {h[0]}: {h[3]}k\n({h[1]}m², {h[2]}b)", (h[1] + 1, h[3] - 15), fontsize=8.5)

# Highlight Top 5 Neighbors
for item in top5_house:
    ax1.scatter([item[1]], [item[3]], s=240, facecolors='none', edgecolors='#e63946', linewidth=2.5, zorder=5)

# Plot predicted house
ax1.scatter([query_house[0]], [y_hat_house], c='#e63946', marker='*', s=380, edgecolors='black', linewidth=1.5,
            label=f'Predicted: 72m² -> {y_hat_house:.1f}k', zorder=6)
ax1.axvline(x=72, color='#e63946', linestyle='--', alpha=0.5)
ax1.axhline(y=y_hat_house, color='#e63946', linestyle='--', alpha=0.5)

ax1.set_title('Weighted KNN Regression: House Price Prediction\n(Area & Beds -> Price in Thousand THB, K=5)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Area (sq.m.)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Price (Thousand Baht)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: 1D Continuous Curve Comparison (Simple vs Weighted Average)
x_curve = np.linspace(45, 105, 200)
y_simple_curve = []
y_weighted_curve = []

for xc in x_curve:
    # 1D distance based on Area
    dists = [abs(xc - h[1]) for h in house_data]
    order = np.argsort(dists)[:k_house]
    # Simple
    y_simple_curve.append(np.mean([prices[i] for i in order]))
    # Weighted
    w_list = [1.0 / (dists[i]**2 + epsilon) for i in order]
    y_weighted_curve.append(sum(w_list[i] * prices[order[i]] for i in range(k_house)) / sum(w_list))

ax2.plot(x_curve, y_simple_curve, color='#457b9d', linestyle='-', linewidth=2, label='KNN Simple Avg Curve (K=5)')
ax2.plot(x_curve, y_weighted_curve, color='#e76f51', linestyle='-', linewidth=2.5, label='KNN Distance-Weighted Curve (K=5)')
ax2.scatter(areas, prices, c='#1d3557', s=80, edgecolors='black', label='Data Points', zorder=4)
ax2.scatter([72], [y_hat_house], c='#e63946', marker='*', s=250, zorder=6, label=f'Target x=72')

ax2.set_title('Continuous Regression Curves Comparison\nSimple vs Distance-Weighted KNN Regression', fontsize=12, fontweight='bold')
ax2.set_xlabel('Area (sq.m.)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Price (Thousand Baht)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/knn_regression_slide.png', dpi=300)
plt.close()
print("Saved figure to figures/knn_regression_slide.png")
