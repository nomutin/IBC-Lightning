"""
モデルの学習.

References
----------
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html

"""

from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """Execute lightning cli."""
    LightningCLI(save_config_callback=None)


if __name__ == "__main__":
    main()
