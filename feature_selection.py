import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === Load Data ===
df = pd.read_csv("SPY_full_processed_data.csv")
df = df.select_dtypes(include=[np.number])

thresh = round(0.9 * df.shape[0])
df = df.dropna(axis=1, thresh=thresh)
df = df.dropna()

# === Set Target ===
target = 'target_direction'
X = df.drop(columns=target).values
Y = df[target].values
input_names = df.drop(columns=target).columns.values
N, D = X.shape

# === Center Inputs and Outputs for Correlation ===
mu_X = X.mean(axis=0)
mu_Y = Y.mean()
Xc = X - mu_X
Yc = Y - mu_Y

Sigma2X = (Xc.T @ Xc) / (N - 1)
SigmaXY = (Xc.T @ Yc) / (N - 1)

theta1hat = np.linalg.inv(Sigma2X + 1e-6*np.eye(D)) @ SigmaXY
theta0hat = mu_Y - mu_X @ theta1hat

Yhat = theta0hat + X @ theta1hat
sigmahat2 = np.sum((Y - Yhat)**2) / (N - D - 1)

var_thetahat = sigmahat2 * np.linalg.inv(Sigma2X + 1e-6*np.eye(D)).diagonal()
var_thetahat = np.clip(var_thetahat, 0, None)
z = abs(stats.norm.ppf((1 - 0.95)/2))
rho = z * np.sqrt(var_thetahat)
significant = (np.abs(theta1hat) - rho) > 0
significant_inputs = input_names[significant]

# === Manual Additions ===
manual_add = {
    'fear_greed_value', 'VIX', 'Volume',
    'volatility_20d', 'volatility_50d',
    'RSI', 'Open', 'Close'
}
final_inputs = set(significant_inputs).union(manual_add)

# === Final dataset ===
data = df[list(final_inputs)].copy()
data["Y"] = df[target].values

# === Train/Validate/Test Split ===
N = len(data)
Ntrain = round(0.7 * N)
Nvalidate = round(0.15 * N)
Ntest = N - Ntrain - Nvalidate
Dtrain = data.iloc[:Ntrain]
Dvalidate = data.iloc[Ntrain:Ntrain+Nvalidate]
Dtest = data.iloc[Ntrain+Nvalidate:]

# === Train function ===
def train(S, D):
    X = D[list(S)].values
    Y = D['Y'].values
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean()
    Xc = X - mu_X
    Yc = Y - mu_Y
    CovX = (Xc.T @ Xc) / (X.shape[0] - 1)
    CovX += 1e-6 * np.eye(CovX.shape[0])  # regularization
    invCovX = np.linalg.inv(CovX)
    theta1 = invCovX @ ((Xc.T @ Yc) / (X.shape[0] - 1))
    theta0 = mu_Y - mu_X @ theta1
    return theta0, theta1

# === Assess function ===
def assess(S, theta0, theta1, D):
    X = D[list(S)].values
    Y = D['Y'].values
    Yhat = theta0 + X @ theta1
    MSE = np.mean((Y - Yhat)**2)
    return MSE

# === Forward stepwise selection ===
def forward_stepwise_selection(all_inputs):
    D = len(all_inputs)
    setD = set(all_inputs)
    setS = [set() for _ in range(D+1)]
    ellk = np.full(D+1, np.inf)
    setS[0] = set()

    for k in range(1, D+1):
        setA = []
        ellkappa = []

        for xp in setD - setS[k-1]:
            candidate = setS[k-1].union({xp})
            try:
                theta0, theta1 = train(candidate, Dtrain)
                ell = assess(candidate, theta0, theta1, Dvalidate)
                setA.append(candidate)
                ellkappa.append(ell)
            except Exception:
                continue

        if not ellkappa:
            break

        best_idx = np.argmin(ellkappa)
        setS[k] = setA[best_idx]
        ellk[k] = ellkappa[best_idx]

    kstar = np.argmin(ellk)
    Sstar = setS[kstar]
    theta0star, theta1star = train(Sstar, Dtrain)
    ellstar = assess(Sstar, theta0star, theta1star, Dtest)
    return ellk, ellstar, kstar, Sstar

# === Run selection and evaluate ===
ellk, ellstar, kstar, Sstar = forward_stepwise_selection(significant_inputs)

# === Merge with manual features ===
Sstar = Sstar.union(manual_add)
theta0, theta1 = train(Sstar, Dtrain)
test_mse = assess(Sstar, theta0, theta1, Dtest)

# === Output ===
print(f"Best k = {kstar}, Validation MSE = {ellk[kstar]:.4f}")
print(f"Test MSE with manual features forced in: {test_mse:.4f}")
print("Final selected features:", sorted(Sstar))
