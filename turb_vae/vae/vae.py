from typing import Tuple

import torch

# from layers import Decoder2d, Encoder2d
from torch import nn

from .layers import Decoder2d, Encoder2d

# from turb_vae.vae.layers import Decoder2d, Encoder2d


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

        self.event_size = loc.shape[-1]
        self.batch_size = loc.shape[0]

    def kl_divergence(self) -> torch.Tensor:
        r"""
        The KL divergence from a standard multivariate normal.
        We take the average over the event size as we will do with the cross entropy loss.
        """

        # mu^T mu
        sqr_mu = self.loc.pow(2).sum(-1)
        
        # the trace of the covariance matrix
        # self.cov_factor.shape = (batch_size, event_size, rank)
        # tr(cov_factor^T cov_factor) = sum_{j=1}^k sum_{i=1}^N(cov_factor *h* cov_factor)_{:,i,j}
        # where *h* is the Hadamard/elementwise product
        tr_cov = self.cov_diag.sum(-1) + (self.cov_factor**2).sum((-1, -2))

        # the log determinant of the covariance matrix
        # this takes advantage of the pre-computed Cholesky decomposition of the capacitance matrix called _capacitance_tril
        # _capacitance_tril.shape = (batch_size, event_size, event_size)
        logdet_sigma = self.cov_diag.log().sum(-1) + 2 * self._capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)

        return 0.5 * (tr_cov + sqr_mu - self.event_size - logdet_sigma) / self.event_size
        

class LowRankVariationalAutoencoder(nn.Module):
    r"""
    The `VariationalAutoencoder` class.

    Args:
        encoder (`Encoder2d`):
            The encoder network.
        decoder (`Decoder2d`):
            The decoder network.
        rank (`int`):
            The rank of the approximate posterior.
        num_particles (`int`, optional):
            The number of particles to sample from the approximate posterior. Default is 1.
        cov_factor_init_scale (`float`, optional):
            The scale to initialize the off-diagonal part of the covariance matrix. Default is 1.
            This scales the output of the encoder elements for cov_factor. 
    """

    def __init__(self, encoder: Encoder2d, decoder: Decoder2d, rank: int, num_particles=1, cov_factor_init_scale = 1.):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rank = rank
        self.num_particles = num_particles
        self.cov_factor_init_scale = cov_factor_init_scale

        assert rank >= 1 and isinstance(rank, int), "The rank must be a positive integer."

        assert (
            encoder.out_channels == decoder.in_channels * (2 + self.rank)
        ), "The output channels of the encoder equal (2 + rank) times the input channels of the decoder."

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, LowRankMultivariateNormal]:
        r"""The forward method of the `VariationalAutoencoder` class."""
        dist_pars = self.encoder(x)
        # dist_pars.shape = (batch_size, out_channels, height, width)

        # pull off the individual params
        loc = dist_pars[..., : self.decoder.in_channels, :, :]  # mu.shape = (batch_size, C, H, W)
        batch_size = loc.shape[0]
        CHW = loc.shape[1:]
        cov_diag = dist_pars[..., self.decoder.in_channels : 2*self.decoder.in_channels, :, :].exp() # logD.shape = (batch_size, C, H, W)
        cov_factor = self.cov_factor_init_scale * dist_pars[..., 2*self.decoder.in_channels :, :, :] # A.shape = (batch_size, C*rank, H, W)

        # reshape the parameters for creating the distribution object
        loc = loc.view(batch_size, -1)  # mu.shape = (batch_size, C*H*W)
        cov_diag = cov_diag.view(batch_size, -1)  # logD.shape = (batch_size, C*H*W)
        cov_factor = cov_factor.view(batch_size, self.rank, -1).permute(0, 2, 1)  # A.shape = (batch_size, C*H*W, rank)

        # the distribution object
        dist = LowRankMultivariateNormal(loc, cov_factor, cov_diag.exp())
        
        
        z = dist.rsample((self.num_particles,))  # z.shape = (num_particles, batch_size, C*H*W)
        # reshape for the decoder
        # we combine batch and particle dimensions because conv2d doesn't allow arbitrary batching on left
        z = z.view(self.num_particles * batch_size, *CHW)
        # z.shape = (num_particles*batch_size, C, H, W)
            
        x_hat = self.decoder(z)  # x_hat.shape = (num_particles*batch_size, C_out, H_out, W_out)
    
        # now separate the batch and particle dimensions
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
    kl = dist.kl_divergence()

    print(kl)
