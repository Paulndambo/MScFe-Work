import numpy as np
import matplotlib.pyplot as plt

def simulate_black_scholes(S0, mu, sigma, T, dt=1/252, n_sim=1):
    """
    Simulates stock returns using the Black-Scholes GBM model.
    
    Parameters:
        S0 (float): Initial stock price
        mu (float): Expected annual return
        sigma (float): Annual volatility
        T (int): Time horizon in years
        dt (float): Time step (default is 1 trading day)
        n_sim (int): Number of simulation paths

    Returns:
        np.ndarray: Simulated stock price paths
    """
    N = int(T / dt)  # Total steps
    t = np.linspace(0, T, N)
    
    # Generate random normal variables for Brownian motion
    W = np.random.standard_normal(size=(N, n_sim))
    
    # Compute stock price paths
    S = np.zeros((N, n_sim))
    S[0] = S0
    
    for i in range(1, N):
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[i])
    
    return t, S

# Parameters
S0 = 100     # Initial stock price
mu = 0.1     # Expected return (10% annual)
sigma = 0.2  # Volatility (20% annual)
T = 1        # Time horizon in years
n_sim = 5    # Number of simulations

# Run simulation
t, S = simulate_black_scholes(S0, mu, sigma, T, n_sim)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, S)
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.title('Black-Scholes Simulated Stock Prices')
plt.grid()
plt.show()
