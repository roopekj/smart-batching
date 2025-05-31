import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from settings import lambda_param, num_docs, seed

# Setting seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Lengths of documents
doc_lengths = [1 + int(x) for x in np.random.exponential(1 / lambda_param, num_docs)]

fontsize = 24
plt.xlabel("Review length", fontsize=fontsize)
plt.ylabel("Density", fontsize=fontsize)
plt.title(
    f"Movie review lengths simulated from an exponential distribution (lambda = {lambda_param})",
    fontsize=fontsize,
)
plt.hist(doc_lengths, bins=50, density=True, alpha=0.6, color="b")
x = np.linspace(0, max(doc_lengths), 100)
pdf = lambda_param * np.exp(-lambda_param * x)
plt.plot(x, pdf, "r-", label="Theoretical PDF")
plt.legend(fontsize=fontsize)
plt.show()
