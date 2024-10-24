"""Lightning module for ibc-lightning."""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from einops.layers.torch import Rearrange
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchgeometry.contrib import SpatialSoftArgmax2d
from tqdm import tqdm
from wandb import Image
from whitecanvas import new_grid
from whitecanvas.canvas import CanvasGrid

from ibc_lightning import IBC


class LitIBC(LightningModule):
    """LigitningModule Wrapper for IBC."""

    def __init__(self) -> None:
        super().__init__()
        state_encoder = nn.Sequential(
            SpatialSoftArgmax2d(),
            Rearrange("b d xy -> b (d xy)"),
        )
        energy_head = nn.Sequential(
            nn.Linear(6 + 2, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )
        self.model = IBC(
            state_encoder=state_encoder,
            energy_head=energy_head,
            upper_bounds=(1.0, 1.0),
            lower_bounds=(-1.0, -1.0),
            inference_samples=2**10,
        )

    def training_step(self, batch: tuple[Tensor, Tensor], *_args: int) -> Tensor:
        """Run training step."""
        observations, actions = batch
        loss = self.model.calc_loss(actions=actions, states=observations)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], *_args: int) -> Tensor:
        """Run validation step."""
        observations, actions = batch
        loss = self.model.calc_loss(actions=actions, states=observations)
        self.log("val_loss", loss)
        return loss


class LogLitIBC(Callback):
    """Log LitIBC Output."""

    def __init__(self, every_n_epochs: int, num_samples: int) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run IBC().predict() and log output."""
        if trainer.current_epoch % self.every_n_epochs != 0 or trainer.current_epoch <= 1:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not isinstance(pl_module, LitIBC):
            return

        for stage in ("train", "val"):
            dataloader = getattr(trainer.datamodule, f"{stage}_dataloader")()  # type: ignore[attr-defined]
            observations, actions = next(iter(dataloader))
            observations = observations[:self.num_samples].to(pl_module.device)
            actions = actions[:self.num_samples].to(pl_module.device)
            prediction_list = []
            for t in tqdm(range(actions.shape[1])):
                prediction = pl_module.model.inference(state=observations[:, t, :])
                prediction_list.append(prediction)
            predictions = torch.stack(prediction_list, dim=1)
            with TemporaryDirectory() as tmp_dir:
                figure_path = Path(tmp_dir) / "prediction.png"
                plot_joint_prediction(target=actions, prediction=predictions)
                plt.savefig(figure_path)
                logger.experiment.log({f"joint_prediction({stage})": Image(str(figure_path))})
                plt.close()
                plt.clf()


def plot_joint_prediction(target: Tensor, prediction: Tensor) -> CanvasGrid:
    """
    関節角度の推移を可視化する.

    Parameters
    ----------
    target : Tensor
        目標関節角度. shape: [B, L, D].
    prediction : Tensor
        予測関節角度. shape: [B, L, D].

    Returns
    -------
    CanvasGrid
    """
    target_array = target.detach().cpu().numpy()
    prediction_array = prediction.detach().cpu().numpy()
    batch_size, _, dim = target_array.shape
    size = (batch_size * 400, dim * 200)
    grid = new_grid(rows=dim, cols=batch_size, backend="matplotlib", size=size)
    for b, d in product(range(batch_size), range(dim)):
        c = grid.add_canvas(d, b)
        c.update_font(family="Noto Mono", size=10)
        c.add_line(target_array[b, :, d], color="blue", name="target")
        c.add_line(prediction_array[b, :, d], color="red", name="prediction")
        if d == 0:
            c.title = f"Batch #{b}"
        if b == 0 and d == 0:
            c.add_legend()
    return grid
