import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GumbelTopK(nn.Module):
    """Differentiable Top-K token selection using Gumbel-Top-K"""
    
    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = k
        self.tau = tau
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_tokens] token scores
        Returns:
            weights: [batch_size, num_tokens] soft selection weights
        """
        batch_size, num_tokens = logits.shape
        
        # Add Gumbel noise
        uniform_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        perturbed_logits = (logits + gumbel_noise) / self.tau
        
        # Softmax to get continuous approximation
        weights = F.softmax(perturbed_logits, dim=-1)
        
        # Optional: Sparse top-k projection (can be disabled for pure continuous)
        if self.training:
            # During training, use continuous weights
            return weights
        else:
            # During inference, use hard top-k
            _, top_k_indices = torch.topk(logits, self.k, dim=-1)
            hard_weights = torch.zeros_like(logits)
            hard_weights.scatter_(-1, top_k_indices, 1.0 / self.k)
            return hard_weights


class TokenScorer(nn.Module):
    """MLP-based token scorer for computing token importance"""
    
    def __init__(self, token_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, num_tokens, token_dim]
        Returns:
            scores: [batch_size, num_tokens]
        """
        return self.scorer(tokens).squeeze(-1)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 改用LayerNorm替代BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalize


def compute_token_average(tokens: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute weighted average of tokens
    Args:
        tokens: [batch_size, num_tokens, token_dim]
        weights: [batch_size, num_tokens]
    Returns:
        averaged: [batch_size, token_dim]
    """
    # Normalize weights to sum to k (approximately)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    normalized_weights = weights / (weight_sum + eps)
    
    # Weighted average
    weighted_tokens = tokens * normalized_weights.unsqueeze(-1)
    return weighted_tokens.sum(dim=1)


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance (1 - cosine_similarity)"""
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance"""
    return torch.sum((x - y) ** 2, dim=-1)
