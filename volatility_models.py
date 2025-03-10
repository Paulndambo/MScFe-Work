import numpy as np
import matplotlib.pyplot as plt

def simulate_local_volatility(S0, mu, T, dt=1/252, n_sim=5):
    """
    Simulates stock prices under the Local Volatility Model.
    
    Parameters:
        S0 (float): Initial stock price
        mu (float): Drift (expected return)
        T (float): Time horizon in years
        dt (float): Time step (default is daily)
        n_sim (int): Number of simulation paths
    
    Returns:
        np.ndarray: Simulated stock price paths
    """
    N = int(T / dt)  # Number of steps
    t = np.linspace(0, T, N)
    
    # Function defining local volatility
    def local_vol(S, t):
        return 0.2 + 0.1 * np.sin(0.1 * S)  # Example function for local volatility
    
    S = np.zeros((N, n_sim))
    S[0] = S0
    
    for j in range(n_sim):
        for i in range(1, N):
            sigma = local_vol(S[i-1], t[i-1])
            dW = np.random.normal(0, np.sqrt(dt))
            S[i, j] = S[i-1, j] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return t, S

# Parameters
S0 = 100
mu = 0.05
T = 1  # 1 year
n_sim = 5

# Simulate
t, S = simulate_local_volatility(S0, mu, T, n_sim=n_sim)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, S)
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.title('Local Volatility Model Simulation')
plt.grid()
plt.show()



def simulate_heston(S0, V0, mu, kappa, theta, xi, rho, T, dt=1/252, n_sim=5):
    """
    Simulates stock prices under the Heston Stochastic Volatility Model.
    
    Parameters:
        S0 (float): Initial stock price
        V0 (float): Initial variance
        mu (float): Drift
        kappa (float): Mean reversion speed
        theta (float): Long-term mean of variance
        xi (float): Volatility of volatility
        rho (float): Correlation between asset and variance
        T (float): Time horizon in years
        dt (float): Time step (default is daily)
        n_sim (int): Number of simulation paths
    
    Returns:
        np.ndarray: Simulated stock price paths
    """
    N = int(T / dt)  # Number of steps
    t = np.linspace(0, T, N)
    
    S = np.zeros((N, n_sim))
    V = np.zeros((N, n_sim))
    
    S[0] = S0
    V[0] = V0
    
    for j in range(n_sim):
        for i in range(1, N):
            Z1, Z2 = np.random.normal(0, 1, 2)  # Generate two correlated random variables
            dW_V = Z1 * np.sqrt(dt)
            dW_S = rho * Z1 * np.sqrt(dt) + np.sqrt(1 - rho**2) * Z2 * np.sqrt(dt)
            
            # Update variance (square of volatility)
            V[i, j] = np.abs(V[i-1, j] + kappa * (theta - V[i-1, j]) * dt + xi * np.sqrt(V[i-1, j]) * dW_V)
            
            # Update stock price
            S[i, j] = S[i-1, j] * np.exp((mu - 0.5 * V[i, j]) * dt + np.sqrt(V[i, j]) * dW_S)
    
    return t, S

# Parameters
S0 = 100
V0 = 0.04  # Initial variance (square of volatility)
mu = 0.05
kappa = 2.0
theta = 0.04
xi = 0.3
rho = -0.5
T = 1
n_sim = 5

# Simulate
t, S = simulate_heston(S0, V0, mu, kappa, theta, xi, rho, T, n_sim=n_sim)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, S)
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.title('Heston Stochastic Volatility Model Simulation')
plt.grid()
plt.show()
