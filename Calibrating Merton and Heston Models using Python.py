import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fft import fft
from numpy import exp, log, pi

def merton_cf(u, T, r, sigma, lam, m, v):
    """
    Merton Jump-Diffusion characteristic function
    """
    omega = r - 0.5 * sigma**2 - lam * (exp(m + 0.5 * v**2) - 1)
    cf = exp(1j * u * omega * T - 0.5 * sigma**2 * u**2 * T +
             lam * T * (exp(1j * u * m - 0.5 * v**2 * u**2) - 1))
    return cf


def heston_cf(u, T, r, v0, kappa, theta, sigma, rho):
    """
    Heston model characteristic function
    """
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * 1j * u - b)**2 + (sigma**2) * (1j * u + u**2))
    g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)
    exp1 = 1j * u * r * T + (a / sigma**2) * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    exp2 = (v0 / sigma**2) * (b - rho * sigma * 1j * u + d) * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
    return exp(exp1 - exp2)


def carr_madan_fft(cf_func, S0, K, T, r, alpha=1.5, N=2**12, eta=0.25, **cf_kwargs):
    logK = np.log(K)
    lambd = 2 * np.pi / (N * eta)
    b = N * lambd / 2
    u = np.arange(N) * eta
    ku = -b + lambd * np.arange(N)

    discount = np.exp(-r * T)
    cf_values = cf_func(u - (alpha + 1) * 1j, T=T, r=r, **cf_kwargs)

    integrand = (
        discount * cf_values
        / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
        * np.exp(1j * u * np.log(S0))
        * np.exp(-1j * u * logK) * eta
    )
    integrand[0] *= 0.5

    fft_values = fft(integrand).real
    call_prices = np.exp(-alpha * ku) * fft_values / np.pi

    idx = np.argmin(np.abs(ku - logK))
    return call_prices[idx]


# Synthetic market data (e.g., from Black-Scholes)
def generate_market_prices(S0, K_list, T, r, sigma_true):
    from scipy.stats import norm
    market_prices = []
    for K in K_list:
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma_true**2) * T) / (sigma_true * np.sqrt(T))
        d2 = d1 - sigma_true * np.sqrt(T)
        call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        market_prices.append(call)
    return np.array(market_prices)


def calibrate_merton(S0, K_list, T, r, market_prices):
    def objective(params):
        sigma, lam, m, v = params
        model_prices = [carr_madan_fft(merton_cf, S0, K, T, r, sigma=sigma, lam=lam, m=m, v=v) for K in K_list]
        return np.mean((np.array(model_prices) - market_prices)**2)

    res = minimize(objective, x0=[0.2, 0.1, -0.1, 0.2], bounds=[(0.01, 1), (0.01, 1), (-1, 1), (0.01, 1)])
    return res


def calibrate_heston(S0, K_list, T, r, market_prices):
    def objective(params):
        v0, kappa, theta, sigma, rho = params
        model_prices = [carr_madan_fft(heston_cf, S0, K, T, r, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho) for K in K_list]
        return np.mean((np.array(model_prices) - market_prices)**2)

    res = minimize(objective, x0=[0.04, 1, 0.04, 0.3, -0.5], bounds=[(0.001, 1), (0.01, 5), (0.001, 1), (0.01, 1), (-0.99, 0.99)])
    return res


S0 = 100
r = 0.05
T = 1.0
K_list = np.linspace(80, 120, 10)
market_prices = generate_market_prices(S0, K_list, T, r, sigma_true=0.2)

res_merton = calibrate_merton(S0, K_list, T, r, market_prices)
res_heston = calibrate_heston(S0, K_list, T, r, market_prices)

print("Merton calibrated params:", res_merton.x)
print("Heston calibrated params:", res_heston.x)
