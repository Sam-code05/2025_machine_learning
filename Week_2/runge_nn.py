import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# Data
rng = np.random.default_rng(42)
N_train = 256; N_val = 256
X_train = rng.uniform(-1.0, 1.0, size=(N_train, 1)); y_train = f(X_train)
X_val = rng.uniform(-1.0, 1.0, size=(N_val, 1)); y_val = f(X_val)
X_grid = np.linspace(-1.0, 1.0, 1000).reshape(-1, 1); y_true_grid = f(X_grid)

# Model
h1, h2 = 32, 32
rng = np.random.default_rng(123)
def glorot(n_in, n_out):
    lim = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-lim, lim, size=(n_in, n_out))
W1 = glorot(1, h1); b1 = np.zeros(h1)
W2 = glorot(h1, h2); b2 = np.zeros(h2)
W3 = glorot(h2, 1);  b3 = np.zeros(1)

params = [W1, b1, W2, b2, W3, b3]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]

def forward(X):
    z1 = X @ W1 + b1; a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2; a2 = np.tanh(z2)
    z3 = a2 @ W3 + b3; yhat = z3
    return yhat, (X, z1, a1, z2, a2, z3, yhat)

def mse(yhat, y):
    return np.mean((yhat - y)**2)

def backward(cache, y):
    X, z1, a1, z2, a2, z3, yhat = cache
    N = X.shape[0]
    dy = 2.0 * (yhat - y) / N
    dW3 = a2.T @ dy; db3 = dy.sum(axis=0)
    da2 = dy @ W3.T; dz2 = da2 * (1.0 - a2**2)
    dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T; dz1 = da1 * (1.0 - a1**2)
    dW1 = X.T @ dz1;  db1 = dz1.sum(axis=0)
    return [dW1, db1, dW2, db2, dW3, db3]

def adam_step(params, grads, m, v, lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
        m_hat = m[i] / (1 - beta1**t); v_hat = v[i] / (1 - beta2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

# Train
epochs = 1200; best_val = np.inf; best_snapshot = None; t = 0
train_losses, val_losses = [], []
for epoch in range(1, epochs + 1):
    yhat, cache = forward(X_train)
    loss = mse(yhat, y_train)
    grads = backward(cache, y_train)
    t += 1
    params = [W1, b1, W2, b2, W3, b3]
    params = adam_step(params, grads, m, v, lr=2e-3, t=t)
    W1, b1, W2, b2, W3, b3 = params

    train_losses.append(loss)
    val_pred, _ = forward(X_val); val_loss = mse(val_pred, y_val)
    val_losses.append(val_loss)
    if val_loss < best_val:
        best_val = val_loss; best_snapshot = [p.copy() for p in params]

# Evaluate
W1, b1, W2, b2, W3, b3 = best_snapshot
y_pred_grid, _ = forward(X_grid)
mse_grid = mse(y_pred_grid, y_true_grid)
max_err_grid = np.max(np.abs(y_pred_grid - y_true_grid))
print(f"MSE on grid: {mse_grid:.6e},  Max error: {max_err_grid:.6e}")

# Plots
plt.figure()
plt.plot(X_grid.squeeze(), y_true_grid.squeeze(), label="True f(x)")
plt.plot(X_grid.squeeze(), y_pred_grid.squeeze(), label="NN prediction")
plt.scatter(X_train.squeeze(), y_train.squeeze(), s=10, alpha=0.3, label="Train samples")
plt.title("Runge function approximation"); plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("runge_true_vs_nn.png", dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(train_losses, label="Train MSE")
plt.plot(val_losses, label="Val MSE")
plt.title("Training/Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
plt.savefig("runge_loss_curves.png", dpi=200, bbox_inches="tight")
