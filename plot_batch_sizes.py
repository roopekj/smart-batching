import pickle

import matplotlib.pyplot as plt
import numpy as np

fontsize = 24
types = {"smart": ["g", 0.3], "hard": ["b", 0.2], "dumb": ["r", 0.1]}
for batch_type, (color, opacity) in types.items():
    with open("batch_sizes/%s.pickle" % batch_type, "rb") as f:
        data = pickle.load(f)
    x = np.arange(len(data))
    plt.plot(x, data, "%s-" % color, label=batch_type)
    plt.fill_between(x, data, alpha=opacity, color=color)


plt.xlabel("Batch", fontsize=fontsize)
plt.ylabel("Matrix dimension", fontsize=fontsize)
plt.title("Matrix dimensions for each batch", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.grid()
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

for i, (batch_type, (color, opacity)) in enumerate(types.items()):
    with open("batch_sizes/%s.pickle" % batch_type, "rb") as f:
        data = pickle.load(f)
    x = np.arange(len(data))
    axes[i].plot(x, data, "%s-" % color, label=batch_type)
    axes[i].fill_between(x, data, alpha=0.3, color=color)

    axes[i].set_ylabel("Matrix dimension", fontsize=fontsize)
    axes[i].set_title(f"Matrix dimensions for {batch_type} batch", fontsize=fontsize)
    axes[i].grid()

axes[-1].set_xlabel("Batch", fontsize=fontsize)
plt.tight_layout()
plt.show()
