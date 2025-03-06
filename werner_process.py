import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=1, N=1000, seed=None):
    """
    Simulates a standard Brownian motion (Wiener process).
    
    Parameters:
    - T: Total time (default 1 year)
    - N: Number of time steps
    - seed: Random seed for reproducibility
    
    Returns:
    - t: Time grid
    - W: Brownian motion path
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N  # Time step
    t = np.linspace(0, T, N + 1)  # Time grid
    dW = np.sqrt(dt) * np.random.randn(N)  # Gaussian increments
    W = np.insert(np.cumsum(dW), 0, 0)  # Cumulative sum with W(0) = 0

    return t, W

# Simulate Brownian motion
T = 1  # Time period
N = 1000  # Number of steps
t, W = brownian_motion(T, N, seed=42)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, W, label="Wiener Process W(t)", color="b")
plt.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
plt.xlabel("Time (t)")
plt.ylabel("W(t)")
plt.title("Standard Brownian Motion (Wiener Process)")
plt.legend()
plt.grid()
plt.show()
