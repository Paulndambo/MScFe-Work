import numpy as np

class BinomialTreeOption:
    def __init__(self, S0, K, T, r, sigma, N, option_type="call"):
        self.S0 = S0          # Initial stock price
        self.K = K            # Strike price
        self.T = T            # Time to maturity
        self.r = r            # Risk-free rate
        self.sigma = sigma    # Volatility
        self.N = N            # Number of time steps
        self.option_type = option_type.lower()  # "call" or "put"

        # Compute binomial model parameters
        self.dt = T / N  # Time step
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # Risk-neutral probability
    
    def build_tree(self):
        """Builds the stock price tree."""
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return stock_tree

    def option_payoff(self, stock_tree):
        """Computes the option payoffs at maturity."""
        payoffs = np.maximum(0, stock_tree[:, -1] - self.K) if self.option_type == "call" \
                  else np.maximum(0, self.K - stock_tree[:, -1])
        return payoffs

    def price_option(self):
        """Uses backward induction to price the option using the binomial tree."""
        stock_tree = self.build_tree()
        option_tree = np.zeros_like(stock_tree)
        option_tree[:, -1] = self.option_payoff(stock_tree)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-self.r * self.dt) * (self.p * option_tree[j, i + 1] + (1 - self.p) * option_tree[j + 1, i + 1])
        
        return option_tree, stock_tree

    def compute_hedging_strategy(self):
        """Computes the delta (hedging ratio) at each step."""
        _, stock_tree = self.price_option()
        option_tree, _ = self.price_option()
        
        delta_hedge = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(i + 1):
                # Delta = (option up - option down) / (stock up - stock down)
                delta_hedge[j, i] = (option_tree[j, i + 1] - option_tree[j + 1, i + 1]) / (stock_tree[j, i + 1] - stock_tree[j + 1, i + 1])

        return delta_hedge

# Example parameters
S0 = 100     # Initial stock price
K = 100      # Strike price
T = 1        # Time to maturity (1 year)
r = 0.05     # Risk-free interest rate
sigma = 0.2  # Volatility
N = 3        # Number of time steps

option = BinomialTreeOption(S0, K, T, r, sigma, N, option_type="call")
option_price_tree, stock_tree = option.price_option()
delta_hedge = option.compute_hedging_strategy()

# Display results
print("Stock Price Tree:\n", stock_tree)
print("\nOption Price Tree:\n", option_price_tree)
print("\nDelta Hedging Strategy:\n", delta_hedge)
