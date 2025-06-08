import math

import torch
from torch import Tensor
from torch.nn import functional as F

from settings import dtype, hidden_dim, num_layers, vocab_size


class WhateverModel:
    def __init__(self):
        self.padding_token_id = -100

        # Embeddings
        self.embeddings = torch.rand(
            [vocab_size, 1, hidden_dim], dtype=dtype, device="cuda"
        )

        # Attention weights
        self.q_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.k_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.v_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.o_proj_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]

        # Linear layers
        self.linear_weights = [
            torch.rand([hidden_dim, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        self.linear_biases = [
            torch.rand([1, hidden_dim], dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]

        # Decision head
        self.decision_head = torch.rand([hidden_dim, 2], dtype=dtype, device="cuda")

    @staticmethod
    def normalize(X: Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        return (X - mean) / (std + 1e-5)

    def embed_documents(self, documents: list[list[int]]) -> tuple[Tensor, Tensor]:
        bs, seq_len = len(documents), len(documents[0])

        X = torch.zeros([bs, seq_len, hidden_dim], dtype=dtype)
        attention_mask = torch.zeros([bs, seq_len], dtype=dtype)

        for i, document in enumerate(documents):
            num_padding = len([a for a in document if a == self.padding_token_id])
            attention_mask[i, :] = torch.cat(
                [
                    torch.zeros([1, seq_len - num_padding]),
                    torch.ones([1, num_padding], dtype=dtype) * float("-inf"),
                ],
                dim=1,
            )

            X[i] = torch.cat(
                [self.embeddings[token] for token in documents[i]], dim=0
            ).to(dtype)
            if num_padding:
                X[i, -num_padding:, :] = torch.zeros([num_padding, hidden_dim])

        return X.to("cuda"), attention_mask.to("cuda")

    def forward(self, documents: list[list[int]]) -> list[bool]:
        X, attention_mask = self.embed_documents(documents)

        attention_mask = attention_mask.unsqueeze(2)
        attention_mask = torch.cat(
            [attention_mask] * attention_mask.shape[1], dim=2
        ).transpose(1, 2)

        for i in range(num_layers):
            # Attention layer
            q_states = X @ self.q_proj_weights[i]
            k_states = X @ self.k_proj_weights[i]
            v_states = X @ self.v_proj_weights[i]

            attn_weights = q_states @ k_states.transpose(1, 2) / math.sqrt(hidden_dim)
            attn_weights += attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights @ v_states

            X = attn_weights @ self.o_proj_weights[i]

            # Linear layer
            X = X @ self.linear_weights[i] + self.linear_biases[i]

            # Normalization so the values don't explode
            X = self.normalize(X)

        # Just take the first token's output as a pseudo "class token"
        cls_tokens = (X @ self.decision_head)[:, 0, :]
        decisions: list[bool] = [(tok[1] >= tok[0]).item() for tok in cls_tokens]  # type: ignore

        return decisions
