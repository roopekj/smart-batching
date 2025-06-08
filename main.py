import random
from time import time

import numpy as np
import torch

from helpers import (
    batch_birdbrained,
    batch_clever,
    batch_industrious,
    batch_smartest,
    generate_documents,
)
from model import WhateverModel
from settings import BatchType, batching_type, lambda_param, num_docs, seed

# Setting seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Lengths of documents
doc_lengths = [1 + int(x) for x in np.random.exponential(1 / lambda_param, num_docs)]

model = WhateverModel()
documents = generate_documents(doc_lengths=doc_lengths)

start = time()
if batching_type == BatchType.smartest:
    sentiments = batch_smartest(model=model, documents=documents)
elif batching_type == BatchType.clever:
    sentiments = batch_clever(model=model, documents=documents)
elif batching_type == BatchType.industrious:
    sentiments = batch_industrious(model=model, documents=documents)
elif batching_type == BatchType.birdbrain:
    sentiments = batch_birdbrained(model=model, documents=documents)
else:
    raise RuntimeError("Invalid batching type")

end = time()

print(sentiments)
print(f"Forward pass took {end - start} seconds.")
