from typing import Tuple

import torch
from torch import nn

from turb_vae.vae.layers import Decoder2d, Encoder2d


class VariationalAutoencoder(nn.Module):
    r"""
    The `VariationalAutoencoder` class.

    Args:
        in_channels (`int`):
            The number of input channels.
        latent_dim (`int`):
            The dimension of the latent space.
        num_blocks_encoder (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderBlock`'s to
            use.
        block_out_channels_encoder (`Tuple[int, ...]`):
            The number of output channels for each block in the encoder.
        num_blocks_decoder (`Tuple[int, ...]`):
            Each value of the tuple represents a Conv2d layer followed by `value` number of `AutoencoderBlock`'s to
            use.
        block_out_channels_decoder (`Tuple[int, ...]`):
            The number of output channels for each block in the decoder.
        upsampling_scaling_factor (`int`):
            The scaling factor to use for upsampling.
        act_fn (`str`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
    """

    def __init__(self, encoder: Encoder2d, decoder: Decoder2d):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert (
            encoder.out_channels == decoder.in_channels
        ), "The output channels of the encoder must match the input channels of the decoder."

    def forward(
        self, x: torch.FloatTensor, sample: bool = True
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        r"""The forward method of the `VariationalAutoencoder` class."""
        mu_logvar = self.encoder(x)
        mu = mu_logvar[..., : self.decoder.in_channels, :, :]
        logvar = mu_logvar[..., self.decoder.in_channels :, :, :]
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(logvar / 2)
        else:
            z = mu
        x_hat = self.decoder(z)

        return x_hat, mu, logvar

    @staticmethod
    def KL_divergence(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.Tensor:
        r"""
        The KL divergence between the approximate posterior and the prior.
        This function normalizes the KL divergence by the number of pixels in the latent
        representation.

        Args:
            mu (`torch.FloatTensor`):
                The mean of the approximate posterior.
            logvar (`torch.FloatTensor`):
                The log-variance of the approximate posterior.

        Returns:
            `torch.FloatTensor`: The KL divergence.
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())



if __name__ == "__main__":
    enc = Encoder2d(1, 16, (1, 3, 3, 3), (64, 64, 64, 64), "relu")
    dec = Decoder2d(16, 1, (3, 3, 3, 1), (64, 64, 64, 64), 2, "relu")

    # test encoder and decoder
    x = torch.randn(3, 1, 8, 8)
    print("Input shape:", x.shape)
    mlv = enc(x)
    print("Encoder output shape:", mlv.shape)
    mu = mlv[..., : dec.in_channels, :, :]
    logvar = mlv[..., dec.in_channels :, :, :]
    z = mu + torch.randn_like(mu) * torch.exp(logvar / 2)
    print("z shape:", z.shape)
    print("Decoder ouput shape:", dec(z).shape)

    # test the autoencoder
    vae = VariationalAutoencoder(enc, dec)
    x_hat, mu, logvar = vae(x)
    print("VAE output shape:", x_hat.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
