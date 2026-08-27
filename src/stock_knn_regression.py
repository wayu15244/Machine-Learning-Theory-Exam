import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Synthetic AAPL Stock Price Data (Simulated for 2024)
# Demonstrating 02_Exercise-KNN-Regression.ipynb
# -------------------------------------------------------------
np.random.seed(42)
days = 120
t = np.linspace(0, 4 * np.pi, days)
# Realistic stock series with trend, cyclical component, and noise
base_price = 175 + 10 * np.sin(t / 2) + 0.15 * np.arange(days)
noise = np.random.normal(0, 1.2, days)
close_prices = base_price + noise

def create_sliding_window_dataset(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

window = 5
X, y = create_sliding_window_dataset(close_prices, window_size=window)

# Train/Test split: 80% train, 20% test
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

def knn_regressor_predict(X_tr, y_tr, X_te, k=3, weighted=False, epsilon=1e-5):
    preds = []
    for x in X_te:
        dists = np.sqrt(np.sum((X_tr - x)**2, axis=1))
        k_indices = np.argsort(dists)[:k]
        if not weighted:
            pred = np.mean(y_tr[k_indices])
        else:
            k_dists = dists[k_indices]
            weights = 1.0 / (k_dists**2 + epsilon)
            pred = np.sum(weights * y_tr[k_indices]) / np.sum(weights)
        preds.append(pred)
    return np.array(preds)

# Test k=3, k=5, and weighted
k3_preds = knn_regressor_predict(X_train, y_train, X_test, k=3, weighted=False)
k5_preds = knn_regressor_predict(X_train, y_train, X_test, k=5, weighted=False)
k5_weighted_preds = knn_regressor_predict(X_train, y_train, X_test, k=5, weighted=True)

mae_k3 = np.mean(np.abs(y_test - k3_preds))
mae_k5 = np.mean(np.abs(y_test - k5_preds))
mae_k5_w = np.mean(np.abs(y_test - k5_weighted_preds))

print(f"KNN Regression Stock Prediction Results (Window={window}):")
print(f"  MAE (k=3, Simple): {mae_k3:.4f}")
print(f"  MAE (k=5, Simple): {mae_k5:.4f}")
print(f"  MAE (k=5, Weighted): {mae_k5_w:.4f}")

# Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.figure(figsize=(12, 5), dpi=300)

train_timeline = range(window, split_idx + window)
test_timeline = range(split_idx + window, len(close_prices))

plt.plot(train_timeline, y_train, color='#1d3557', label='Train Close Price', linewidth=1.8)
plt.plot(test_timeline, y_test, color='#457b9d', label='Actual Test Close Price', linewidth=2)
plt.plot(test_timeline, k3_preds, color='#e63946', linestyle='--', label=f'KNN (k=3) Pred (MAE={mae_k3:.2f})', linewidth=1.8)
plt.plot(test_timeline, k5_weighted_preds, color='#2a9d8f', linestyle='-.', label=f'KNN (k=5, Wtd) Pred (MAE={mae_k5_w:.2f})', linewidth=2)

plt.title(f'Stock Price Prediction with KNN Regression (Sliding Window={window} days)', fontsize=13, fontweight='bold')
plt.xlabel('Trading Day', fontsize=11, fontweight='bold')
plt.ylabel('Close Price (USD)', fontsize=11, fontweight='bold')
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/stock_knn_regression.png', dpi=300)
plt.close()
print("Saved figure to figures/stock_knn_regression.png")
