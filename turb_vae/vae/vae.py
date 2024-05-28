from typing import Tuple

import torch

if __name__ == "__main__":
    from layers import Decoder2d, Encoder2d
else:
    from .layers import Decoder2d, Encoder2d

from torch import nn


def prod(t: Tuple[int | float, ...]) -> int | float:
    r"""The product of a tuple of numbers."""

    p = 1
    for i in t:
        p *= i

    return p


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
        return -0.5 * torch.mean(1 + self.scale.log() - self.loc.pow(2) - self.scale)


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

    def __init__(
        self, loc: torch.Tensor, cov_factor: torch.Tensor, cov_diag: torch.Tensor
    ):
        super().__init__(loc, cov_factor, cov_diag)

        self.event_size = loc.shape[-1]
        self.batch_size = loc.shape[0]

    def kl_divergence(self) -> torch.Tensor:
        r"""
        The KL divergence from a standard multivariate normal.
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
        logdet_sigma = self.cov_diag.log().sum(
            -1
        ) + 2 * self._capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)

        return 0.5 * (tr_cov + sqr_mu - self.event_size - logdet_sigma)


class VAE(nn.Module):
    r"""
    The `LowRankVAE` class.

    Args:
        encoder_kwargs (`dict`):
            The arguments for the encoder network.
        decoder_kwargs (`dict`):
            The arguments for the decoder network.
        embed_dim (`int`):
            The size of the latent dimension.
        input_size (`Tuple[int, int]`):
            The shape of the input tensor (H, W)
        rank (`int`):
            The rank of the approximate posterior.
        num_particles (`int`, optional):
            The number of particles to sample from the approximate posterior. Default is 1.
        cov_factor_init_scale (`float`, optional):
            The scale to initialize the off-diagonal part of the covariance matrix. Default is 1.
            This scales the output of the encoder elements for cov_factor.
            Has no effect for rank=0.
    """

    def __init__(
        self,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        embed_dim: int,
        input_size: Tuple[int, int],
        rank: int,
        is_conditional: bool,
        num_particles: int = 1,
        cov_factor_init_scale: float = 1.0,
    ):
        super().__init__()

        assert rank >= 0 and isinstance(
            rank, int
        ), "The rank must be a non-negative integer."

        self.encoder = Encoder2d(**encoder_kwargs)
        self.decoder = Decoder2d(**decoder_kwargs)
        assert (
            self.encoder.downsample_factor == self.decoder.upsample_factor
        ), "The encoder and decoder must have the same downsample/upsample factor."

        self.embed_dim = embed_dim
        self.input_size = input_size
        self.rank = rank
        self.is_conditional = is_conditional
        self.num_particles = num_particles
        self.cov_factor_init_scale = cov_factor_init_scale

        # encoder linear projection
        self._enc_output_shape = (
            self.encoder.out_channels,
            self.input_size[0] // self.encoder.downsample_factor,
            self.input_size[1] // self.encoder.downsample_factor,
        )
        self.fc_encode = nn.Sequential(
            nn.Tanh(),
            nn.Linear(
                in_features=int(prod(self._enc_output_shape)),
                out_features=self.embed_dim * (2 + self.rank),
            ),
        )

        # decoder linear projection
        self._dec_input_shape = (
            self.decoder.in_channels,
            self.input_size[0] // self.encoder.downsample_factor,
            self.input_size[1] // self.encoder.downsample_factor,
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(
                in_features=self.embed_dim
                if not is_conditional
                else self.embed_dim + 1,
                out_features=int(prod(self._dec_input_shape)),
            ),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, L0: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, LowRankMultivariateNormal]:
        r"""
        The forward method of the `VariationalAutoencoder` class.

        Args:
            x (`torch.Tensor`): (batch_size, C|C-1, H, W)
                The input tensor. When `is_conditional` is `True`, x
                has one less channel than expected by the encoder because
                the L0 value is concatenated to the input tensor.
            L0 (`torch.tTensor`): (batch_size, 1)
                The L0 value for the conditional VAE. If `None`, this is ignored.
        """
        dist = self.encode(x, L0)

        z = dist.rsample(
            (self.num_particles,)  # type: ignore
        )  # (num_particles, batch_size, embed_dim)

        x_hat = self.decode(z, L0)

        return x_hat, dist

    def encode(self, x: torch.Tensor, L0: torch.Tensor | None = None):
        batch_size = x.shape[0]

        if self.is_conditional:
            assert (
                L0 is not None
            ), "The L0 value must be provided for a conditional VAE."

            # concatenate the L0 value to the input tensor
            L0_catx = L0[:, :, None, None].expand(-1, -1, *self.input_size)
            x = torch.cat([x, L0_catx], dim=1)
        # x.shape = (batch_size, C, H, W)

        h = self.encoder(x)  # (batch_size, *_enc_output_shape)
        h = h.view(batch_size, -1)  # (batch_size, prod(_enc_output_shape))
        dist_pars = self.fc_encode(h)  # (batch_size, embed_dim*(2+rank))

        # pull off the individual params
        loc = dist_pars[..., : self.embed_dim]  # (batch_size, embed_dim)
        diag = dist_pars[
            ..., self.embed_dim : 2 * self.embed_dim
        ].exp()  # (batch_size, embed_dim)
        # we only pull off cov_factor for rank>0
        
        # create the distribution object
        if self.rank == 0:
            dist = DiagonalMultivariateNormal(loc=loc, scale=diag)
        else:
            cov_factor = (
                self.cov_factor_init_scale * dist_pars[..., 2 * self.embed_dim :]
            )  # (batch_size, embed_dim*rank)
            cov_factor = cov_factor.view(
                batch_size, self.embed_dim, self.rank
            )  # (batch_size, embed_dim, rank)
            dist = LowRankMultivariateNormal(loc, cov_factor, diag)

        return dist

    def decode(self, z: torch.Tensor, L0: torch.Tensor | None = None):
        batch_size = z.shape[1]

        if self.is_conditional:
            # concatenate the L0 value to the latent variables
            L0_catz = L0[None, :, :].expand(self.num_particles, -1, -1)  # type: ignore
            z = torch.cat([z, L0_catz], dim=-1)
        # z.shape = (num_particles, batch_size, embed_dim|embed_dim+1)

        # project for the decoder
        h_dec = self.fc_decode(z)  # (num_particles, batch_size, prod(dec_input_shape))

        # reshape for the decoder combining batch and particle dimensions
        h_dec = h_dec.view(
            self.num_particles * batch_size, *self._dec_input_shape
        )  # (num_particles*batch_size, *_dec_input_shape)

        # decode
        x_hat = self.decoder(h_dec)  # (num_particles*batch_size, C, *input_shape)

        # now separate the batch and particle dimensions
        x_hat = x_hat.view(
            self.num_particles, batch_size, *x_hat.shape[1:]
        )  # (num_particles, batch_size, C, *input_shape)

        return x_hat


if __name__ == "__main__":
    batch_size = 5

    import sys

    sys.path.append("turb_vae")
    from config import LightningModelConfig as model_cfg

    vae = VAE(**model_cfg.vae_config)  # type: ignore

    print("Num pars:", sum(p.numel() for p in vae.parameters()))

    x = torch.randn(batch_size, 1, *model_cfg.vae_config["input_size"])  # type: ignore
    L0 = torch.ones(batch_size, 1)
    x_hat_proj, dist_proj = vae(x, L0)
    print("VAE output shape:", x_hat_proj.shape)
    kl_proj = dist_proj.kl_divergence()
    print(kl_proj)
