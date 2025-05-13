import numpy as np
from scipy.optimize import minimize

def simulate_CIR(kappa, theta, sigma, r0, T, N, seed=42):
    np.random.seed(seed)
    dt = T / N
    rates = [r0]
    for _ in range(N):
        rt = rates[-1]
        dr = kappa * (theta - rt) * dt + sigma * np.sqrt(max(rt, 0)) * np.sqrt(dt) * np.random.normal()
        rates.append(rt + dr)
    return np.array(rates)

def cir_zero_coupon_bond_price(r, kappa, theta, sigma, T):
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    numerator = 2 * h * np.exp((kappa + h) * T / 2)
    denominator = 2 * h + (kappa + h) * (np.exp(h * T) - 1)
    A = (numerator / denominator)**(2 * kappa * theta / sigma**2)
    B = 2 * (np.exp(h * T) - 1) / denominator
    return A * np.exp(-B * r)

def calibrate_cir(T_list, r_market, r0):
    def objective(params):
        kappa, theta, sigma = params
        model_yields = []
        for T, r_obs in zip(T_list, r_market):
            P = cir_zero_coupon_bond_price(r0, kappa, theta, sigma, T)
            y_model = -np.log(P) / T
            model_yields.append(y_model)
        return np.mean((np.array(model_yields) - np.array(r_market))**2)

    res = minimize(objective, x0=[0.5, 0.05, 0.1], bounds=[(0.01, 3), (0.01, 0.2), (0.001, 1)])
    return res
