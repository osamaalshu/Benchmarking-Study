#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyCELoss(nn.Module):
    """
    Proxy cross-entropy over class proxies.
    - emb: (N, D) pixel embeddings (L2-normalized inside)
    - proxies: (C, D) learnable class proxies (L2-normalized inside)
    - targets: (N,) class indices in [0..C-1]; use ignore_index to skip pixels
    """

    def __init__(self, temperature: float = 0.07, ignore_index: int = -100):
        super().__init__()
        self.tau = temperature
        self.ignore_index = ignore_index

    def forward(self, emb: torch.Tensor, proxies: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # emb: (N, D), proxies: (C, D), targets: (N,)
        if emb.numel() == 0:
            return emb.sum() * 0.0

        emb = F.normalize(emb, dim=1)
        proxies = F.normalize(proxies, dim=1)

        logits = emb @ proxies.t() / self.tau  # (N, C)
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        return loss


