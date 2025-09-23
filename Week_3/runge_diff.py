import numpy as np
import matplotlib.pyplot as plt

def f(x): return 1.0 / (1.0 + 25.0 * x**2)
def fprime(x): return -(50.0 * x) / (1.0 + 25.0 * x**2)**2

rng = np.random.default_rng(42)
N_train = 256; N_val = 256
X_train = rng.uniform(-1.0, 1.0, size=(N_train, 1)); y_train = f(X_train); yprime_train = fprime(X_train)
X_val = rng.uniform(-1.0, 1.0, size=(N_val, 1)); y_val = f(X_val); yprime_val = fprime(X_val)
X_grid = np.linspace(-1.0, 1.0, 1000).reshape(-1, 1)
y_true_grid = f(X_grid); yprime_true_grid = fprime(X_grid)

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

def backward_from_dy(cache, dy):
    X, z1, a1, z2, a2, z3, yhat = cache
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

def nn_and_derivative(X):  # analytic derivative for evaluation
    yhat, cache = forward(X)
    Xc, z1, a1, z2, a2, z3, _ = cache
    dy = np.ones_like(yhat)
    da2 = dy @ W3.T
    dz2 = da2 * (1.0 - a2**2)
    da1 = dz2 @ W2.T
    dz1 = da1 * (1.0 - a1**2)
    dx  = dz1 @ W1.T
    return yhat, dx

def nn_derivative_central(X, h=1e-3):
    y_plus, _ = forward(X + h)
    y_minus, _ = forward(X - h)
    return (y_plus - y_minus) / (2.0*h), y_plus, y_minus

epochs = 1200; lambda_deriv = 1.0; h_fd = 1e-3; t = 0
train_func_losses=[]; train_deriv_losses=[]; val_func_losses=[]; val_deriv_losses=[]
best_val = np.inf; best_snapshot=None

for epoch in range(1, epochs+1):
    yhat, cache = forward(X_train)
    dfdx_fd, y_plus, y_minus = nn_derivative_central(X_train, h=h_fd)
    func_err = yhat - y_train; deriv_err = dfdx_fd - yprime_train
    L_func = np.mean(func_err**2); L_deriv = np.mean(deriv_err**2)
    N = X_train.shape[0]
    dy_at_X = (2.0/N) * func_err
    coeff = (2.0/N) * lambda_deriv * deriv_err * (1.0/(2.0*h_fd))
    dy_at_Xplus  = coeff
    dy_at_Xminus = -coeff
    _, cache_plus  = forward(X_train + h_fd)
    _, cache_minus = forward(X_train - h_fd)
    grads_X      = backward_from_dy(cache,       dy_at_X)
    grads_Xplus  = backward_from_dy(cache_plus,  dy_at_Xplus)
    grads_Xminus = backward_from_dy(cache_minus, dy_at_Xminus)
    grads = [gX + gP + gM for gX,gP,gM in zip(grads_X, grads_Xplus, grads_Xminus)]
    t += 1
    params = [W1, b1, W2, b2, W3, b3]
    params = adam_step(params, grads, m, v, lr=2e-3, t=t)
    W1, b1, W2, b2, W3, b3 = params
    train_func_losses.append(L_func); train_deriv_losses.append(L_deriv)
    yhat_val, _ = forward(X_val)
    dfdx_fd_val, _, _ = nn_derivative_central(X_val, h=h_fd)
    val_func_losses.append(np.mean((yhat_val - y_val)**2))
    val_deriv_losses.append(np.mean((dfdx_fd_val - yprime_val)**2))
    val_total = val_func_losses[-1] + lambda_deriv*val_deriv_losses[-1]
    if val_total < best_val:
        best_val = val_total; best_snapshot = [p.copy() for p in params]

W1, b1, W2, b2, W3, b3 = best_snapshot
y_pred_grid, yprime_pred_grid = nn_and_derivative(X_grid)
mse_f = np.mean((y_pred_grid - y_true_grid)**2)
max_f = np.max(np.abs(y_pred_grid - y_true_grid))
mse_fp = np.mean((yprime_pred_grid - yprime_true_grid)**2)
max_fp = np.max(np.abs(yprime_pred_grid - yprime_true_grid))
print(f"f: MSE={mse_f:.6e}, Max={max_f:.6e} | f': MSE={mse_fp:.6e}, Max={max_fp:.6e}")

plt.figure(); plt.plot(X_grid.squeeze(), y_true_grid.squeeze(), '--', linewidth=2.5, label="True f(x)")
plt.plot(X_grid.squeeze(), y_pred_grid.squeeze(), label="NN prediction",)
plt.legend(); plt.title("Runge: true vs NN"); plt.savefig("runge_sobolev_fx.png", dpi=200, bbox_inches="tight")

plt.figure(); plt.plot(X_grid.squeeze(), yprime_true_grid.squeeze(), '--', linewidth=2.5, label="True f'(x)")
plt.plot(X_grid.squeeze(), yprime_pred_grid.squeeze(), label="NN dŷ/dx")
plt.legend(); plt.title("Runge derivative: true vs NN"); plt.savefig("runge_sobolev_fprime.png", dpi=200, bbox_inches="tight")

plt.figure(); 
plt.plot(train_func_losses, label="Train func MSE")
plt.plot(val_func_losses, label="Val func MSE")
plt.plot(train_deriv_losses, label="Train deriv MSE")
plt.plot(val_deriv_losses, label="Val deriv MSE")
plt.legend(); plt.title("Loss curves"); plt.savefig("runge_sobolev_losses.png", dpi=200, bbox_inches="tight")
