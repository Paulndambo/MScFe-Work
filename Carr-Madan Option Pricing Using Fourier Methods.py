import numpy as np
from scipy.fft import fft
from numpy import exp, log, pi

def bs_characteristic_function(u, T, r, sigma):
    """
    Characteristic function for log-price under Black-Scholes
    """
    return np.exp(1j * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)

def carr_madan_option_price_fft(S0, K, T, r, sigma, alpha=1.5, N=2**12, eta=0.25):
    """
    Carr-Madan Fourier pricing method for European Call options
    """
    # Log of the strike
    logK = np.log(K)

    # Setup FFT grid
    lambd = 2 * np.pi / (N * eta)  # spacing for log-strikes
    b = N * lambd / 2
    u = np.arange(N) * eta  # frequencies
    ku = -b + lambd * np.arange(N)  # log-strikes

    # Discount factor
    discount = np.exp(-r * T)

    # Characteristic function values
    cf_values = bs_characteristic_function(u - (alpha + 1) * 1j, T, r, sigma)

    # FFT integrand
    integrand = (
        discount
        * cf_values
        / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
        * np.exp(1j * u * (log(S0)) )
    )
    integrand *= np.exp(-1j * u * logK) * eta
    integrand[0] *= 0.5  # Simpson's rule correction

    # Perform FFT
    fft_values = fft(integrand).real

    # Find strike closest to desired K
    k_index = int((logK + b) / lambd)
    call_price = np.exp(-alpha * ku) * fft_values / np.pi

    return ku[k_index], call_price[k_index]
