"""Tests for `core.py`."""

import pytest
from torch import Tensor

from ibc_lightning import IBC
from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    DummyEncoder,
    DummyHead,
)


@pytest.fixture()
def ibc() -> IBC:
    """ダミーの IBC インスタンス."""
    return IBC(
        obs_encoder=DummyEncoder(),
        energy_head=DummyHead(),
        upper_bounds=tuple(1.0 for _ in range(ACTION_SIZE)),
        lower_bounds=tuple(-1.0 for _ in range(ACTION_SIZE)),
        train_samples=32,
        inference_samples=64,
    )


def test__shared_step_single(
    ibc: IBC,
    observations: Tensor,
    actions: Tensor,
) -> None:
    """`shared_step` の簡単なテストケース."""
    batch = (observations, actions)
    loss_dict = ibc.shared_step(batch=batch)
    assert "loss" in loss_dict
    assert loss_dict["loss"].shape == ()


def test__shared_step_timeseries(
    ibc: IBC,
    observations_timeseries: Tensor,
    actions_timeseries: Tensor,
) -> None:
    """`shared_step` の簡単なテストケース(時系列)."""
    batch = (observations_timeseries, actions_timeseries)
    loss_dict = ibc.shared_step(batch=batch)
    assert "loss" in loss_dict
    assert loss_dict["loss"].shape == ()


def test__predict_step(ibc: IBC, observations: Tensor) -> None:
    """`predict_step` の簡単なテストケース."""
    actions = ibc.predict_step(observations)
    assert actions.shape == (BATCH_SIZE, ACTION_SIZE)
    assert actions.min() >= -1.0
    assert actions.max() <= 1.0


def test__training_step(
    ibc: IBC,
    observations: Tensor,
    actions: Tensor,
) -> None:
    """`training_step` の簡単なテストケース."""
    loss_dict = ibc.training_step(batch=(observations, actions))
    assert "loss" in loss_dict
    assert loss_dict["loss"].shape == ()


def test__validation_step(
    ibc: IBC,
    observations: Tensor,
    actions: Tensor,
) -> None:
    """`validation_step` の簡単なテストケース."""
    loss_dict = ibc.validation_step(batch=(observations, actions))
    assert "val_loss" in loss_dict
    assert loss_dict["val_loss"].shape == ()
