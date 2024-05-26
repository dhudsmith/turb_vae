import torch

from turb_vae.vae.vae import LowRankVAE


def test_projected_vae():
    rank = 7
    num_particles = 10
    batch_size = 5

    # Create the projected VAE
    proj_enc_kwargs = {
        "in_channels": 1,
        "out_channels": 2 * (2 + rank),
        "input_shape": (1, 3, 3),
        "hidden_dims": (32,) * 3,
        "activation": "relu"
    }
    proj_dec_kwargs = {
        "in_channels": 2,
        "out_channels": 1,
        "output_shape": (3, 3, 1),
        "hidden_dims": (32,) * 3,
        "activation": "relu"
    }
    proj_vae = LowRankVAE(
        encoder_kwargs=proj_enc_kwargs,
        decoder_kwargs=proj_dec_kwargs,
        embed_dim=10,
        input_size=(16, 16),
        rank=rank,
        num_particles=num_particles
    )

    # Generate random input data
    x = torch.randn(batch_size, 1, 16, 16)

    # Test forward pass
    x_hat_proj, dist_proj = proj_vae(x)
    assert x_hat_proj.shape == (num_particles, batch_size, 1, 16, 16)

    # Test KL divergence calculation
    kl_proj = dist_proj.kl_divergence()
    assert kl_proj.shape == (num_particles, batch_size)