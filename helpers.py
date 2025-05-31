import pickle
from random import choices

import numpy as np
import torch
from torch import Tensor

from model import WhateverModel
from settings import batch_size, hidden_dim, num_docs, vocab_size


def generate_documents(doc_lengths):
    documents = []
    for i in range(num_docs):
        numtokens = doc_lengths[i]

        new_tokens = choices(range(vocab_size), k=numtokens)

        documents.append(new_tokens)

    return documents


def pad_documents(model: WhateverModel, documents: list[list[int]]) -> list[list[int]]:
    max_tokens = max([len(doc) for doc in documents])

    output = documents.copy()

    for i in range(len(documents)):
        num_padding = max_tokens - len(documents[i])
        output[i].extend([model.padding_token_id] * num_padding)

    return output


def batch_dumb(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    documents = pad_documents(model, documents)
    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments


def batch_hard(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments


def batch_smart(model: WhateverModel, documents: list[list[int]]):
    sentiments = []
    documents.sort(key=lambda x: len(x))

    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments
