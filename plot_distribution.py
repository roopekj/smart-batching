import matplotlib.pyplot as plt
import numpy as np

lambda_param = 5e-4
num_samples = 10000
random_values = np.random.exponential(1 / lambda_param, num_samples)

plt.hist(random_values, bins=30, density=True, alpha=0.6, color="b")

x = np.linspace(0, max(random_values), 10000)
pdf = lambda_param * np.exp(-lambda_param * x)
plt.plot(x, pdf, "r-", label="Theoretical PDF")

plt.xlabel("Value")
plt.ylabel("Density")
plt.title(f"Exponential Distribution (lambda = {lambda_param})")
plt.legend()
plt.grid()
plt.show()
