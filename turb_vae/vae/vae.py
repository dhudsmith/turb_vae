from typing import Tuple

import torch

# from turb_vae.vae.layers import Decoder2d, Encoder2d
from layers import Decoder2d, Encoder2d
from torch import nn


class DiagonalMultivariateNormal(torch.distributions.Normal):
    r"""
    The `DiagonalMultivariateNormal` class.

    Args:
        loc (`torch.Tensor`):
            The mean of the distribution.
        scale (`torch.Tensor`):
            The standard deviation of the distribution.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__(loc, torch.diag_embed(scale))

    def kl_divergence(self) -> torch.Tensor:
        r"""The KL divergence from a standard multivariate normal"""
        -0.5 * torch.mean(1 + self.scale.log() - self.loc.pow(2) - self.scale)

class LowRankMultivariateNormal(torch.distributions.LowRankMultivariateNormal):
    r"""
    The `LowRankMultivariateNormal` class.

    Args:
        loc (`torch.Tensor`):
            The mean of the distribution.
        cov_diag (`torch.Tensor`):
            The diagonal component of the covariance matrix.
        cov_factor (`torch.Tensor`):
            The factor of the off-diagonal part of the covariance matrix.
    """

    def __init__(self, loc: torch.Tensor, cov_factor: torch.Tensor, cov_diag: torch.Tensor):
        super().__init__(loc, cov_factor, cov_diag)

    def kl_divergence(self) -> torch.Tensor:
        r"""The KL divergence from a standard multivariate normal"""
        raise NotImplementedError("The kl divergence method is not implemented yet.")



class LowRankVariationalAutoencoder(nn.Module):
    r"""
    The `VariationalAutoencoder` class.

    Args:
        encoder (`Encoder2d`):
            The encoder network.
        decoder (`Decoder2d`):
            The decoder network.
        rank (`int`, optional):
            The rank of the approximate posterior. Default is 0.
    """

    def __init__(self, encoder: Encoder2d, decoder: Decoder2d, rank: int, num_particles=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rank = rank
        self.num_particles = num_particles

        assert rank >= 1 and isinstance(rank, int), "The rank must be a positive integer."

        assert (
            encoder.out_channels == decoder.in_channels * (2 + self.rank)
        ), "The output channels of the encoder equal (2 + rank) times the input channels of the decoder."

    def forward(
        self, x: torch.FloatTensor, sample: bool = True
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        r"""The forward method of the `VariationalAutoencoder` class."""
        dist_pars = self.encoder(x)
        # dist_pars.shape = (batch_size, out_channels, height, width)

        # pull off the individual params
        mu = dist_pars[..., : self.decoder.in_channels, :, :]  # mu.shape = (batch_size, C, H, W)
        batch_size = mu.shape[0]
        CHW = mu.shape[1:]
        logD = dist_pars[..., self.decoder.in_channels : 2*self.decoder.in_channels, :, :] # logD.shape = (batch_size, C, H, W)
        A = dist_pars[..., 2*self.decoder.in_channels :, :, :] # A.shape = (batch_size, C*rank, H, W)

        # reshape the parameters for creating the distribution object
        mu = mu.view(batch_size, -1)  # mu.shape = (batch_size, C*H*W)
        logD = logD.view(batch_size, -1)  # logD.shape = (batch_size, C*H*W)
        A = A.view(batch_size, rank, -1).permute(0, 2, 1)  # A.shape = (batch_size, C*H*W, rank)

        # the distribution object
        dist = LowRankMultivariateNormal(mu, A, logD.exp())
        
        if sample:
            z = dist.rsample((self.num_particles,))  # z.shape = (num_particles, batch_size, C*H*W)
        else:
            z = mu

        # reshape for the decoder
        # we combine batch and particle dimensions because conv2d doesn't allow arbitrary batching on left
        z = z.view(self.num_particles * batch_size, *CHW)
        # z.shape = (num_particles*batch_size, C, H, W)
        
        # now separate the batch and particle dimensions
        x_hat = self.decoder(z)  # x_hat.shape = (num_particles*batch_size, C_out, H_out, W_out)
        x_hat = x_hat.view(self.num_particles, batch_size, *x_hat.shape[1:]) # x_hat.shape = (num_particles, batch_size, C_out, H_out, W_out)

        return x_hat, dist


if __name__ == "__main__":
    rank = 7
    num_particles = 10
    batch_size = 5

    enc = Encoder2d(1, 2*(2+rank), (1, 3, 3), (32,)*3, "relu")
    dec = Decoder2d(2, 1, (3, 3, 1), (32,)*3, 2, "relu")

    # test encoder and decoder
    x = torch.randn(batch_size, 1, 16, 16)
    print("Input shape:", x.shape)
    mlv = enc(x)
    print("Encoder output shape:", mlv.shape)
    mu = mlv[..., : dec.in_channels, :, :]
    logvar = mlv[..., dec.in_channels : 2*dec.in_channels, :, :]
    z = mu + torch.randn_like(mu) * torch.exp(logvar / 2)
    print("z shape:", z.shape)
    print("Decoder ouput shape:", dec(z).shape)

    # test the autoencoder
    vae = LowRankVariationalAutoencoder(enc, dec, rank, num_particles=num_particles)
    x_hat, dist = vae(x)
    print("VAE output shape:", x_hat.shape)

    # test the KL divergence
    # kl = dist.kl_divergence()

    print(sum(p.numel() for p in vae.parameters()))
