"""テストの共通設定."""

import pytest
import torch
from torch import Tensor, nn

BATCH_SIZE, SEQ_LEN = 4, 8
CHANNELS, HEIGHT, WIDTH = 3, 64, 64
ACTION_DIM, STATE_DIM, EMBED_DIM = 4, 16, 50


class DummyObsEncoder(nn.Module):
    """
    ダミーの観測のエンコーダ.

    [B, C, H, W] -> [B, E] へ変換するモデル.
    """

    def forward(self, x: Tensor) -> Tensor:
        """入力に関係なく適切な形状のテンソルを返す."""
        batch_size, _, _, _ = x.shape
        return torch.rand(batch_size, EMBED_DIM)


class DummyStateEncoder(nn.Module):
    """
    ダミーの状態のエンコーダ.

    [B, S] -> [B, E] へ変換するモデル.
    """

    def forward(self, x: Tensor) -> Tensor:
        """入力に関係なく適切な形状のテンソルを返す."""
        batch_size, _ = x.shape
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


@pytest.fixture
def observations() -> Tensor:
    """観測."""
    return torch.rand(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def observations_timeseries() -> Tensor:
    """観測時系列."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def actions() -> Tensor:
    """行動."""
    return torch.rand(BATCH_SIZE, ACTION_DIM)


@pytest.fixture
def actions_timeseries() -> Tensor:
    """行動時系列."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_DIM)


@pytest.fixture
def states() -> Tensor:
    """状態."""
    return torch.rand(BATCH_SIZE, STATE_DIM)


@pytest.fixture
def states_timeseries() -> Tensor:
    """状態時系列."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, STATE_DIM)
