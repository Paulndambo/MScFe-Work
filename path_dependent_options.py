import numpy as np

class PathDependentOption:
    def __init__(self, S0, K, T, r, sigma, N, option_type="asian_call"):
        self.S0 = S0  # Initial stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.N = N  # Steps in binomial tree
        self.option_type = option_type.lower()  # "asian_call" or "asian_put"

        # Binomial model parameters
        self.dt = T / N  # Time step
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability

    def build_tree(self):
        """Builds stock price tree and keeps track of path averages."""
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        avg_tree = np.zeros((self.N + 1, self.N + 1))  # Tracks average stock price

        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
                avg_tree[j, i] = (self.S0 + stock_tree[j, i]) / 2  # Approximate running average

        return stock_tree, avg_tree

    def option_payoff(self, avg_tree):
        """Computes the option payoff based on the average price."""
        if self.option_type == "asian_call":
            return np.maximum(avg_tree[:, -1] - self.K, 0)
        elif self.option_type == "asian_put":
            return np.maximum(self.K - avg_tree[:, -1], 0)

    def price_option(self):
        """Uses backward induction to price the Asian option."""
        stock_tree, avg_tree = self.build_tree()
        option_tree = np.zeros_like(stock_tree)
        option_tree[:, -1] = self.option_payoff(avg_tree)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                avg_price = (avg_tree[j, i] * (i + 1) + stock_tree[j, i]) / (i + 2)  # Update running average
                option_tree[j, i] = np.exp(-self.r * self.dt) * (
                    self.p * option_tree[j, i + 1] + (1 - self.p) * option_tree[j + 1, i + 1]
                )

        return option_tree, stock_tree

    def compute_hedging_strategy(self):
        """Computes delta (hedging ratio) at each step."""
        _, stock_tree = self.price_option()
        option_tree, _ = self.price_option()

        delta_hedge = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(i + 1):
                delta_hedge[j, i] = (option_tree[j, i + 1] - option_tree[j + 1, i + 1]) / (
                    stock_tree[j, i + 1] - stock_tree[j + 1, i + 1]
                )

        return delta_hedge

# Example parameters
S0 = 100     # Initial stock price
K = 100      # Strike price
T = 1        # Time to maturity
r = 0.05     # Risk-free rate
sigma = 0.2  # Volatility
N = 3        # Steps in binomial tree

asian_option = PathDependentOption(S0, K, T, r, sigma, N, option_type="asian_call")
option_price_tree, stock_tree = asian_option.price_option()
delta_hedge = asian_option.compute_hedging_strategy()

# Display results
print("Stock Price Tree:\n", stock_tree)
print("\nOption Price Tree:\n", option_price_tree)
print("\nDelta Hedging Strategy:\n", delta_hedge)
