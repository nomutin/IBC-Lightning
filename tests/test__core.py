"""Tests for `core.py`."""

import pytest
from torch import Tensor

from ibc_lightning import IBC
from tests.conftest import (
    ACTION_DIM,
    BATCH_SIZE,
    DummyHead,
    DummyObsEncoder,
    DummyStateEncoder,
)


@pytest.fixture
def ibc() -> IBC:
    """ダミーの IBC インスタンス(行動-観測)."""
    return IBC(
        state_encoder=DummyObsEncoder(),
        energy_head=DummyHead(),
        upper_bounds=tuple(1.0 for _ in range(ACTION_DIM)),
        lower_bounds=tuple(-1.0 for _ in range(ACTION_DIM)),
        train_samples=32,
        inference_samples=64,
    )


@pytest.fixture
def ibc_state() -> IBC:
    """ダミーの IBC インスタンス(行動-状態)."""
    return IBC(
        state_encoder=DummyStateEncoder(),
        energy_head=DummyHead(),
        upper_bounds=tuple(1.0 for _ in range(ACTION_DIM)),
        lower_bounds=tuple(-1.0 for _ in range(ACTION_DIM)),
        train_samples=32,
        inference_samples=64,
    )


def test__calc_loss_single(
    ibc: IBC,
    observations: Tensor,
    actions: Tensor,
) -> None:
    """`calc_loss` の簡単なテストケース."""
    loss = ibc.calc_loss(actions=actions, states=observations)
    assert loss.shape == ()


def test__calc_loss_timeseries(
    ibc: IBC,
    observations_timeseries: Tensor,
    actions_timeseries: Tensor,
) -> None:
    """`calc_loss` の簡単なテストケース(時系列)."""
    loss = ibc.calc_loss(
        actions=actions_timeseries,
        states=observations_timeseries,
    )
    assert loss.shape == ()


def test__shared_step_state(
    ibc_state: IBC,
    states: Tensor,
    actions: Tensor,
) -> None:
    """`calc_loss` の簡単なテストケース(状態-行動)."""
    loss = ibc_state.calc_loss(actions=actions, states=states)
    assert loss.shape == ()


def test__calc_loss_state_timeseries(
    ibc_state: IBC,
    states_timeseries: Tensor,
    actions_timeseries: Tensor,
) -> None:
    """`calc_loss` の簡単なテストケース(状態-行動)."""
    loss = ibc_state.calc_loss(
        actions=actions_timeseries,
        states=states_timeseries,
    )
    assert loss.shape == ()


def test__predict_step(ibc: IBC, observations: Tensor) -> None:
    """`predict_step` の簡単なテストケース."""
    actions = ibc.inference(observations)
    assert actions.shape == (BATCH_SIZE, ACTION_DIM)
    assert actions.min() >= -1.0
    assert actions.max() <= 1.0
