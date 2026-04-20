"""Connector modules between audio encoder and LLM."""

import torch
import torch.nn as nn


class LinearConnector(nn.Module):
    """Simple linear projection from audio embedding dim to LLM dim."""

    def __init__(self, audio_dim: int, llm_dim: int):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)

    def forward(self, x):
        return self.linear(x)


class LinearPoolConnector(nn.Module):
    """Linear projection with temporal pooling (average every k frames)."""

    def __init__(self, audio_dim: int, llm_dim: int, k: int = 5):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.k = k

    def forward(self, x):
        # x: (B, T, D) -> pool every k frames -> (B, T//k, D) -> project
        B, T, D = x.shape
        # Pad T to be divisible by k
        pad_len = (self.k - T % self.k) % self.k
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
        x = x.reshape(B, -1, self.k, D).mean(dim=2)
        return self.linear(x)


class CNNConnector(nn.Module):
    """1D CNN connector for downsampling and projecting audio features."""

    def __init__(self, audio_dim: int, llm_dim: int, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(audio_dim, llm_dim, kernel_size=k, stride=k)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d -> (B, llm_dim, T') -> (B, T', llm_dim)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


def get_connector(connector_name: str, audio_dim: int, llm_dim: int, k: int = 5) -> nn.Module:
    """Factory for connector modules.

    Args:
        connector_name: One of "linear", "linear-pool", "cnn".
        audio_dim: Audio encoder output dimension.
        llm_dim: LLM input embedding dimension.
        k: Pooling/stride factor for linear-pool and cnn connectors.

    Returns:
        An nn.Module connector.
    """
    name = connector_name.lower().replace("_", "-")
    if name == "linear":
        return LinearConnector(audio_dim, llm_dim)
    elif name == "linear-pool":
        return LinearPoolConnector(audio_dim, llm_dim, k)
    elif name == "cnn":
        return CNNConnector(audio_dim, llm_dim, k)
    else:
        raise ValueError(f"Unknown connector: {connector_name}. Choose from: linear, linear-pool, cnn")
