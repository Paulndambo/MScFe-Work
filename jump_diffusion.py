import numpy as np
import matplotlib.pyplot as plt

def simulate_jump_diffusion(S0, mu, sigma, lamb, mu_J, sigma_J, T, dt=1/252, n_sim=5):
    """
    Simulates stock prices using the Merton Jump-Diffusion Model.
    
    Parameters:
        S0 (float): Initial stock price
        mu (float): Expected return (drift)
        sigma (float): Volatility of stock returns
        lamb (float): Average number of jumps per year (Poisson rate)
        mu_J (float): Mean of jump size
        sigma_J (float): Standard deviation of jump size
        T (float): Time horizon in years
        dt (float): Time step (default is daily)
        n_sim (int): Number of simulation paths
    
    Returns:
        np.ndarray: Simulated stock price paths
    """
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)
    
    S = np.zeros((N, n_sim))
    S[0] = S0
    
    for j in range(n_sim):
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion
            dN = np.random.poisson(lamb * dt)  # Poisson jump occurrence
            
            # Jump size
            J = np.exp(np.random.normal(mu_J, sigma_J)) if dN > 0 else 1
            
            # Stock price update with jump component
            S[i, j] = S[i-1, j] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW) * J
    
    return t, S

# Parameters
S0 = 100       # Initial stock price
mu = 0.05      # Drift (5% annual return)
sigma = 0.2    # Volatility (20%)
lamb = 1       # One jump per year on average
mu_J = -0.2    # Mean jump size (-20%)
sigma_J = 0.1  # Jump size volatility (10%)
T = 1          # Time horizon in years
n_sim = 5      # Number of simulations

# Run simulation
t, S = simulate_jump_diffusion(S0, mu, sigma, lamb, mu_J, sigma_J, T, n_sim)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, S)
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.title('Jump-Diffusion Model Simulation')
plt.grid()
plt.show()
