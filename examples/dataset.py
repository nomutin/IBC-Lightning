"""IBCのテスト用の簡単なデータセット."""

import torch
from einops import repeat
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def generete_actions(batch_size: int, num_points: int, noise_std: float) -> Tensor:
    """
    円運動の行動データ[batch_size, num_point, 2]を生成する.

    Parameters
    ----------
    batch_size : int
        バッチサイズ.
    num_points : int
        サンプリングする点数.
    noise_std : float
        ノイズの標準偏差.

    Returns
    -------
    Tensor
        行動データ. shape: [batch_size, num_points, 2]
    """
    t = torch.linspace(0, 2 * torch.pi, num_points)
    x = torch.sin(t).mul(0.8)
    y = torch.cos(t).mul(0.8)
    data = torch.stack((x, y), dim=1)
    data = repeat(data, "N D -> B N D", B=batch_size)
    noise = torch.randn_like(data).mul(noise_std)
    return data.add(noise)


def generate_observations(actions: Tensor, size: int = 64) -> Tensor:
    """
    円運動の観測データ[batch_size, num_point, 3, size, size]を生成する.

    Parameters
    ----------
    actions : Tensor
        行動データ. shape: [batch_size, num_point, 2]
    size : int
        画像のサイズ.

    Returns
    -------
    imgs : Tensor
        画像データ. shape: [batch_size, num_point, 3, size, size]

    """
    batch_size, num_points = actions.shape[:2]
    actions = actions.add(1).mul(size / 2).clamp(0, size - 1)
    imgs = torch.zeros((batch_size, num_points, 3, size, size))
    x, y = actions[:, :, 0].long(), actions[:, :, 1].long()
    for t in range(num_points):
        imgs[:, t, :, x[:, t], y[:, t]] = 1
    return imgs


class CircleDataset(Dataset[tuple[Tensor, Tensor]]):
    """円運動のデータセット."""

    def __init__(self, action: Tensor, observation: Tensor) -> None:
        super().__init__()
        self.action = action
        self.observation = observation

    def __len__(self) -> int:
        """
        データ数.

        Returns
        -------
        int
            データ数.
        """
        return len(self.action)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        データを取得する.

        Parameters
        ----------
        idx : int
            データのインデックス.

        Returns
        -------
        tuple[Tensor, Tensor]
            データ.
        """
        return self.observation[idx, :-1], self.action[idx, 1:]


class CircleDataModule(LightningDataModule):
    """
    円運動のDataModule.

    Parameters
    ----------
    train_size : int
        総学習データ数.
    val_size : int
        総検証データ数.
    num_points : int
        各データの点数.
    batch_size : int
        学習・検証時のバッチサイズ.
    """

    def __init__(
        self,
        *,
        train_size: int,
        val_size: int,
        num_points: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.num_points = num_points
        self.batch_size = batch_size

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        """データセットのセットアップ."""
        train_action = generete_actions(self.train_size, self.num_points, 0.05)
        train_observation = generate_observations(train_action)
        self.train_dataset = CircleDataset(action=train_action, observation=train_observation)

        val_action = generete_actions(self.val_size, self.num_points, 0.05)
        val_observation = generate_observations(val_action)
        self.validation_dataset = CircleDataset(action=val_action, observation=val_observation)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """
        学習用のDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, Tensor]]
            学習用のDataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """
        検証用のDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, Tensor]]
            検証用のDataLoader.
        """
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
        )
