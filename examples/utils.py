"""補助関数・クラス."""

from torch import Tensor, nn


class ModuleList(nn.Module):
    """Listを引数にとるnn.Sequential(instanitiate用)."""

    def __init__(self, modules: list[nn.Module]) -> None:
        super().__init__()
        self.seq = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """順伝播."""
        return self.seq(x)
