from random import choices

from model import WhateverModel
from settings import batch_size, num_docs, vocab_size


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


def batch_birdbrained(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    documents = pad_documents(model, documents)
    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments


def batch_industrious(model: WhateverModel, documents: list[list[int]]):
    sentiments = []

    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments


def batch_clever(model: WhateverModel, documents: list[list[int]]):
    sentiments = []
    documents.sort(key=lambda x: len(x))

    for batch_start in range(0, num_docs, batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

    return sentiments


def batch_smartest(model: WhateverModel, documents: list[list[int]]):
    sentiments = []
    documents.sort(key=lambda x: len(x))

    max_tokens_per_batch = 21000
    max_length_diff_within_batch = 8

    batch_start = 0
    curr_index = 0
    while curr_index < num_docs:
        curr_batch_tokens = 0
        curr_batch_min_length = len(documents[batch_start])

        batch = []
        while curr_index < num_docs:
            next_doc_len = len(documents[curr_index])
            if (
                next_doc_len - curr_batch_min_length > max_length_diff_within_batch
                or curr_batch_tokens + next_doc_len > max_tokens_per_batch
            ):
                break

            curr_batch_tokens += len(documents[curr_index])
            curr_index += 1

        batch = documents[batch_start:curr_index]
        batch = pad_documents(model, batch)

        # Forward pass of the model
        sentiments.extend(model.forward(batch))

        batch_start = curr_index

    return sentiments
