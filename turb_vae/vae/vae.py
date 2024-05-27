from typing import Tuple

import torch
from torch import nn

from .layers import Decoder2d, Encoder2d

# from .layers import Decoder2d, Encoder2d

# from turb_vae.vae.layers import Decoder2d, Encoder2d


def prod(t: Tuple[int|float, ...]) -> int|float:
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

class LowRankVAE(nn.Module):
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
    """

    def __init__(self, 
                 encoder_kwargs: dict, 
                 decoder_kwargs: dict, 
                 embed_dim: int, 
                 input_size: Tuple[int, int], 
                 rank: int, 
                 num_particles=1, 
                 cov_factor_init_scale = 1.):
        super().__init__()

        assert rank >= 1 and isinstance(rank, int), "The rank must be a positive integer."

        self.encoder = Encoder2d(**encoder_kwargs)
        self.decoder = Decoder2d(**decoder_kwargs)
        assert self.encoder.downsample_factor == self.decoder.upsample_factor, "The encoder and decoder must have the same downsample/upsample factor."

        self.embed_dim = embed_dim
        self.input_size = input_size
        self.rank = rank
        self.num_particles = num_particles
        self.cov_factor_init_scale = cov_factor_init_scale

        # encoder linear projection
        self._enc_output_shape = (
            self.encoder.out_channels, 
            self.input_size[0] // self.encoder.downsample_factor, 
            self.input_size[1] // self.encoder.downsample_factor
            )
        self.fc_encode =  nn.Sequential(
            nn.Tanh(),
            nn.Linear(
                in_features = prod(self._enc_output_shape), 
                out_features = self.embed_dim * (2 + self.rank)
            ),
        )

        # decoder linear projection
        self._dec_input_shape = (
            self.decoder.in_channels,
            self.input_size[0] // self.encoder.downsample_factor,
            self.input_size[1] // self.encoder.downsample_factor
            )
        self.fc_decode = nn.Sequential(
            nn.Linear(
                in_features = self.embed_dim, 
                out_features = prod(self._dec_input_shape)
            ),
            nn.Tanh()
        )
        

        

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, LowRankMultivariateNormal]:
        r"""The forward method of the `VariationalAutoencoder` class."""
        batch_size = x.shape[0]

        h = self.encoder(x)  # (batch_size, *_enc_output_shape)
        h = h.view(batch_size, -1)  # (batch_size, prod(_enc_output_shape))
        dist_pars = self.fc_encode(h)  # (batch_size, embed_dim*(2+rank))

        # pull off the individual params
        loc = dist_pars[..., :self.embed_dim]  # (batch_size, embed_dim)
        cov_diag = dist_pars[..., self.embed_dim : 2*self.embed_dim].exp()  # (batch_size, embed_dim)
        cov_factor = self.cov_factor_init_scale * dist_pars[..., 2*self.embed_dim:]  # (batch_size, embed_dim*rank)
        cov_factor = cov_factor.view(batch_size, self.embed_dim, self.rank)  # (batch_size, embed_dim, rank)

        # create the distribution object
        dist = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        
        # sample the latent variables
        z = dist.rsample((self.num_particles,))  # (num_particles, batch_size, embed_dim)

        # project for the decoder
        h_dec = self.fc_decode(z) # (num_particles, batch_size, prod(dec_input_shape))

        # reshape for the decoder combining batch and particle dimensions
        h_dec = h_dec.view(self.num_particles * batch_size, *self._dec_input_shape)  # (num_particles*batch_size, *_dec_input_shape)
            
        # decode
        x_hat = self.decoder(h_dec)  # (num_particles*batch_size, C, *input_shape)
    
        # now separate the batch and particle dimensions
        x_hat = x_hat.view(self.num_particles, batch_size, *x_hat.shape[1:]) # (num_particles, batch_size, C, *input_shape)

        return x_hat, dist

class CondLowRankVAE(nn.Module):
    r"""
    The `CondLowRankVAE` class.

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
    """

    def __init__(self, 
                 encoder_kwargs: dict, 
                 decoder_kwargs: dict, 
                 embed_dim: int, 
                 input_size: Tuple[int, int], 
                 rank: int, 
                 num_particles=1, 
                 cov_factor_init_scale = 1.):
        super().__init__()

        assert rank >= 1 and isinstance(rank, int), "The rank must be a positive integer."

        self.encoder = Encoder2d(**encoder_kwargs)
        self.decoder = Decoder2d(**decoder_kwargs)
        assert self.encoder.downsample_factor == self.decoder.upsample_factor, "The encoder and decoder must have the same downsample/upsample factor."

        self.embed_dim = embed_dim
        self.input_size = input_size
        self.rank = rank
        self.num_particles = num_particles
        self.cov_factor_init_scale = cov_factor_init_scale

        # encoder linear projection
        self._enc_output_shape = (
            self.encoder.out_channels, 
            self.input_size[0] // self.encoder.downsample_factor, 
            self.input_size[1] // self.encoder.downsample_factor
            )
        self.fc_encode =  nn.Sequential(
            nn.Tanh(),
            nn.Linear(
                in_features = prod(self._enc_output_shape), 
                out_features = self.embed_dim * (2 + self.rank)
            ),
        )

        # decoder linear projection
        self._dec_input_shape = (
            self.decoder.in_channels,
            self.input_size[0] // self.encoder.downsample_factor,
            self.input_size[1] // self.encoder.downsample_factor
            )
        self.fc_decode = nn.Sequential(
            nn.Linear(
                in_features = self.embed_dim + 1,  # we add one dimension for the conditioning variable
                out_features = prod(self._dec_input_shape)
            ),
            nn.Tanh()
        )
        
    def forward(
        self, x: torch.FloatTensor, L0: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, LowRankMultivariateNormal]:
        r"""The forward method of the `VariationalAutoencoder` class."""
        batch_size = x.shape[0]

        # concatenate L0 along the channel dimension
        L0_img = L0[:, :, None, None].expand(1, 1, *self.input_size)  # (batch_size, 1, *input_size)
        x = torch.cat([x, L0_img], dim=1)  # (batch_size, 2, *input_size)

        h = self.encoder(x)  # (batch_size, *_enc_output_shape)
        h = h.view(batch_size, -1)  # (batch_size, prod(_enc_output_shape))
        dist_pars = self.fc_encode(h)  # (batch_size, embed_dim*(2+rank))

        # pull off the individual params
        loc = dist_pars[..., :self.embed_dim]  # (batch_size, embed_dim)
        cov_diag = dist_pars[..., self.embed_dim : 2*self.embed_dim].exp()  # (batch_size, embed_dim)
        cov_factor = self.cov_factor_init_scale * dist_pars[..., 2*self.embed_dim:]  # (batch_size, embed_dim*rank)
        cov_factor = cov_factor.view(batch_size, self.embed_dim, self.rank)  # (batch_size, embed_dim, rank)

        # create the distribution object
        dist = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        
        # sample the latent variables
        z = dist.rsample((self.num_particles,))  # (num_particles, batch_size, embed_dim)

        # concatenate the conditioning variable
        L0_z = L0[None, None, :].expand(self.num_particles, batch_size, 1)  # (num_particles, batch_size, L0_dim)

        # project for the decoder
        h_dec = self.fc_decode(z) # (num_particles, batch_size, prod(dec_input_shape))

        # reshape for the decoder combining batch and particle dimensions
        h_dec = h_dec.view(self.num_particles * batch_size, *self._dec_input_shape)  # (num_particles*batch_size, *_dec_input_shape)
            
        # decode
        x_hat = self.decoder(h_dec)  # (num_particles*batch_size, C, *input_shape)
    
        # now separate the batch and particle dimensions
        x_hat = x_hat.view(self.num_particles, batch_size, *x_hat.shape[1:]) # (num_particles, batch_size, C, *input_shape)

        return x_hat, dist


if __name__ == "__main__":
    batch_size = 5

    # Test the projected vae
    vae_config = dict( 
        encoder_kwargs = dict(
            in_channels = 1,
            out_channels = 16,
            num_blocks = (1, 5, 5, 5, 5),
            block_out_channels = (96,) * 5,
            act_fn = 'relu'
        ),
        decoder_kwargs = dict(
            in_channels = 4,
            out_channels = 1,
            num_blocks = (3, 3, 3, 1),
            block_out_channels = (64,) * 4,
            act_fn = 'relu'
        ),
        rank = 5,
        embed_dim = 512,
        input_size = (64, 64),
        num_particles = 5,
        cov_factor_init_scale = 1e-3,    
    )

    proj_vae = LowRankVAE(**vae_config)

    print("Num pars:", sum(p.numel() for p in proj_vae.parameters()))

    x = torch.randn(batch_size, 1, *vae_config['input_size'])
    x_hat_proj, dist_proj = proj_vae(x)
    print("Projected VAE output shape:", x_hat_proj.shape)
    kl_proj = dist_proj.kl_divergence()
    print(kl_proj)