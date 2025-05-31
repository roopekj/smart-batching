from enum import Enum

import torch


class BatchType(Enum):
    smart = 1
    hard = 2
    dumb = 3


hidden_dim = 4096
vocab_size = 65535
num_docs = 1000
batch_size = 8
num_layers = 70
lambda_param = 2e-2
seed = 42
dtype = torch.float16
batching_type = BatchType.dumb
