from typing import Tuple

import torch
from torch import nn

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class BaseAutoencoderBlock(nn.Module):
    """
    Base Autoencoder block used in [`Autoencoder`]. It is a mini residual module consisting of plain conv + ReLU blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.FloatTensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse = get_activation(act_fn)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.fuse(self.conv(x) + self.skip(x))


class AutoencoderBlock2d(BaseAutoencoderBlock):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.FloatTensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, act_fn=act_fn
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            self.fuse,
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            self.fuse,
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )

        self.skip = (
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.fuse(self.conv(x) + self.skip(x))


class Encoder2d(nn.Module):
    r"""
    The `Encoder` layer.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        act_fn: str,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]

            if i == 0:
                layers.append(
                    nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
                )
            else:
                layers.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )

            for _ in range(num_block):
                layers.append(AutoencoderBlock2d(num_channels, num_channels, act_fn))

        # output conv
        layers.append(
            nn.Conv2d(
                block_out_channels[-1], out_channels, kernel_size=3, padding=1
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        return self.layers(x)


class Decoder2d(nn.Module):
    r"""
    The `Decoder` layer

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of output channels.
        num_blocks (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderTinyBlock`'s to
            use.
        block_out_channels (`Tuple[int, ...]`):
            The number of output channels for each block.
        upsampling_scaling_factor (`int`):
            The scaling factor to use for upsampling.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                layers.append(AutoencoderBlock2d(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        return self.layers(x)
