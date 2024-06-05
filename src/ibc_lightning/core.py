"""
Implicit Behavioral Cloning.

Glossary
--------
B : バッチサイズ
N : サンプリングの次元
D : 行動次元
E : 観測embedding
C, H, W : 観測のチャンネル, 高さ, 幅

References
----------
* https://arxiv.org/abs/2109.00137
* https://github.com/kevinzakka/ibc
* https://github.com/ALRhub/d3il
"""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as tf
from einops import pack, rearrange, repeat
from lightning import LightningModule
from torch import Tensor, arange, nn

DataGroup: TypeAlias = tuple[Tensor, Tensor]
LossDict: TypeAlias = dict[str, Tensor]


class IBC(LightningModule):
    """
    Implicit Behavioral Cloning(Derivative Free Optimizer).

    Parameters
    ----------
    obs_encoder : nn.Module
        観測のエンコーダ.
        [B, C, H, W] -> [*, E] へ変換するモデルに限る.
    energy_head : nn.Module
        観測・行動からエネルギーを予測するモデル.
        [B, E + D] -> [*, 1] へ変換するモデルに限る.
    lower_bounds : tuple[float, ...]
        各次元の行動の下限. shape: [D]
    upper_bounds : tuple[float, ...] 各次元の行動の上限. shape: [D]
    train_samples : int
        学習時のサンプリング数.
    inference_samples : int
        予測時のサンプリング数.
    """

    def __init__(
        self,
        *,
        obs_encoder: nn.Module,
        energy_head: nn.Module,
        upper_bounds: tuple[float, ...],
        lower_bounds: tuple[float, ...],
        train_samples: int = 2**5,
        inference_samples: int = 2**14,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.energy_head = energy_head
        self.lower_bounds = Tensor(lower_bounds)
        self.upper_bounds = Tensor(upper_bounds)
        self.train_samples = train_samples
        self.inference_samples = inference_samples

    def forward(self, actions: Tensor, observations: Tensor) -> Tensor:
        """
        エネルギーを予測する.

        Parameters
        ----------
        actions : Tensor
            負例込みの行動. shape: [B, N, D]
        observation : Tensor
            観測. shape: [B, C, H, W]

        Returns
        -------
        energy : Tensor
            エネルギー. shape: [B, N]
        """
        obs_embed = self.obs_encoder(observations)
        obs_embed = repeat(obs_embed, "B E -> B N E", N=actions.shape[1])
        energy = self.energy_head(torch.cat([obs_embed, actions], dim=-1))
        return rearrange(energy, "B N 1 -> B N")  # type: ignore[no-any-return]

    def sample(self, batch_size: int, num_samples: int) -> Tensor:
        """
        正例・負例込みでサンプリングする.

        Parameters
        ----------
        batch_size : int
            サンプリングするバッチサイズ.
        num_samples : int
            サンプリングする数.

        Returns
        -------
        samples : Tensor
            サンプル. shape: [batch_size, num_samples, D]
        """
        samples = torch.rand(batch_size, num_samples, len(self.lower_bounds))
        samples = samples.mul(self.upper_bounds - self.lower_bounds)
        samples = samples.add(self.lower_bounds)
        return samples.to(self.device)

    def shuffle(
        self,
        positive: Tensor,
        negatives: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        正例と負例をシャッフルする.

        Parameters
        ----------
        positive : Tensor
            正例の行動. shape: [B, D]
        negatives : Tensor
            負例の行動. shape: [B, N, D]

        Returns
        -------
        Tensor
            シャッフル後の行動. shape: [B, N+1, D]
        Tensor
            正例のインデックス. shape: [B]

        """
        positive = repeat(positive, "B D -> B 1 D")
        samples = torch.cat([positive, negatives], dim=-2)
        permutation = torch.rand(samples.shape[:-1]).argsort(dim=1)
        samples = samples[arange(samples.shape[0])[..., None], permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)
        return samples, ground_truth

    def shared_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        """
        学習ステップ.

        Parameters
        ----------
        batch : DataGroup
            observation: Tensor
                観測. shape: [*B, C, H, W]
            action: Tensor
                行動. shape: [*B, D]

        Returns
        -------
        loss : LossDict
            損失.
        """
        observations, action = batch

        observations, _ = pack([observations], "* C H W")
        action, _ = pack([action], "* D")

        negatives = self.sample(
            batch_size=action.shape[0],
            num_samples=self.train_samples,
        )
        actions, ground_truth = self.shuffle(
            positive=action,
            negatives=negatives,
        )
        energy = self.forward(
            actions=actions,
            observations=observations,
        )
        loss = tf.cross_entropy(
            input=energy.mul(-1),
            target=ground_truth,
        )

        return {"loss": loss}

    def predict_step(
        self,
        observation: Tensor,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.50,
        num_iters: int = 3,
    ) -> Tensor:
        """
        予測ステップ.

        Parameters
        ----------
        observation : Tensor
            観測. shape: [B, C, H, W]
        noise_scale : float
            ノイズの初期倍率.
        noise_shrink : float
            反復ごとのノイズの倍率.
        num_iters : int
            反復数.

        Returns
        -------
        action : Tensor
            最もエネルギーが低い行動. shape: [B, D]

        """
        batch_size = observation.shape[0]
        samples = self.sample(
            batch_size=batch_size,
            num_samples=self.inference_samples,
        )

        for _ in range(num_iters):
            energies = self.forward(
                observations=observation,
                actions=samples,
            )
            probs = tf.softmax(energies.mul(-1.0), dim=-1)
            idxs = torch.multinomial(
                input=probs,
                num_samples=self.inference_samples,
                replacement=True,
            )

            samples = samples[torch.arange(batch_size)[..., None], idxs]
            samples = torch.randn_like(samples).mul(noise_scale).add(samples)
            samples = samples.clamp(
                min=self.lower_bounds,
                max=self.upper_bounds,
            )

            noise_scale *= noise_shrink

        energies = self.forward(
            observations=observation,
            actions=samples,
        )
        probs = tf.softmax(energies.mul(-1.0), dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(batch_size), best_idxs, :]

    def training_step(self, batch: DataGroup, **_: str) -> LossDict:
        """学習ステップ."""
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: DataGroup, **_: int) -> LossDict:
        """検証ステップ."""
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict
