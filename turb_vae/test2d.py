import os

import numpy as np
import torch
import torch.nn as nn
from config import DataConfig as data_cfg
from data_generation.dataset import VonKarmanXY
from matplotlib import pyplot as plt
from train2d import VAETrainer

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAETrainer.load_from_checkpoint("checkpoints/epoch=0-step=125000-v2.ckpt").to(
    device
)
model.vae.num_particles = 1
# export for turbulence removal project
# torch.save(model.vae.decoder.state_dict(), "/home/hudson/code/turbulence_model/models/turb_vae/decoder_kl_1.pt")

# test the reconstructions
upsample = nn.Upsample(**data_cfg.upsample_kwargs)  # type: ignore
L0_vals = np.logspace(**data_cfg.L0_vals_logspace_kwargs)  # type: ignore
dataset = VonKarmanXY(
    **data_cfg.test_dataset_kwargs,  # type: ignore
    L0_vals=L0_vals,
    tfms=upsample,
)
dataset.num_samples = 10

# remove files in plots/reconstructions
file_list = os.listdir("plots/reconstructions/")
for file_name in file_list:
    file_path = os.path.join("plots/reconstructions/", file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

# remove files in plots/generations
file_list = os.listdir("plots/generations/")
for file_name in file_list:
    file_path = os.path.join("plots/generations/", file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

# test reconstructions
for ix, (n, L0) in enumerate(dataset):
    print(ix)
    n = n[None]  # (batch_size, C, *input_shape)
    L0 = L0[None, None]  # (batch_size, 1)
    n_hat, dist = model.vae(n.to(device), L0.to(device))

    # plot n and n_hat side by side
    n = n[0, 0].cpu().numpy()
    n_hat = n_hat[0, 0].cpu().numpy()
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    im1 = axs[0].imshow(n, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(n_hat.squeeze(), cmap="gray")
    axs[1].set_title("Reconstructed")
    fig.tight_layout()
    plt.savefig(f"plots/reconstructions/L0={L0.item():0.2f}_{ix}.png")

    # generate a random n at the specified L0
    z = torch.randn(1, 1, model.vae.embed_dim)  # (num_particles, batch_size, embed_dim)
    x_hat = model.vae.decode(z.to(device), L0.to(device))
    plt.matshow(x_hat.squeeze().cpu().numpy(), cmap="gray")
    plt.title(f"Random sample at L0={L0.item():0.2f}")
    plt.tight_layout()
    plt.savefig(f"plots/generations/L0={L0.item():0.2f}_{ix}.png")
