import random
from time import time

import numpy as np
import torch

from helpers import batch_dumb, batch_hard, batch_smart, generate_documents
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
if batching_type == BatchType.smart:
    sentiments = batch_smart(model=model, documents=documents)
elif batching_type == BatchType.hard:
    sentiments = batch_hard(model=model, documents=documents)
elif batching_type == BatchType.dumb:
    sentiments = batch_dumb(model=model, documents=documents)
else:
    raise RuntimeError("Invalid batching type")

end = time()

print(sentiments)
print(f"Forward pass took {end - start} seconds.")
