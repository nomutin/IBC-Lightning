# ruff: noqa: PLR6301
"""テストの共通設定."""

import pytest
import torch
from torch import Tensor, nn

BATCH_SIZE, SEQ_LEN = 4, 8
CHANNELS, HEIGHT, WIDTH = 3, 64, 64
ACTION_SIZE, EMBED_DIM = 4, 16


class DummyEncoder(nn.Module):
    """
    ダミーの観測のエンコーダ.

    [B, C, H, W] -> [B, E] へ変換するモデル.
    """

    def forward(self, x: Tensor) -> Tensor:
        """入力に関係なく適切な形状のテンソルを返す."""
        batch_size, _, _, _ = x.shape
        return torch.rand(batch_size, EMBED_DIM)


class DummyHead(nn.Module):
    """
    ダミーのエネルギー予測器.

    [B, N, A] -> [B, 1] へ変換するモデル.
    """

    def forward(self, x: Tensor) -> Tensor:
        """入力に関係なく適切な形状のテンソルを返す."""
        batch_size, num_samples, _ = x.shape
        return torch.rand(batch_size, num_samples, 1)


@pytest.fixture()
def observations() -> Tensor:
    """4次元の観測."""
    return torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture()
def observations_timeseries() -> Tensor:
    """5次元の観測."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture()
def actions() -> Tensor:
    """2次元の行動."""
    return torch.rand(BATCH_SIZE, ACTION_SIZE)


@pytest.fixture()
def actions_timeseries() -> Tensor:
    """3次元の行動."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_SIZE)
