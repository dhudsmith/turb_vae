import matplotlib.pyplot as plt
import numpy as np
import torch
from data_generation.dataset import VonKarmanXY
from torch import nn

from turb_vae.train2d import VAETrainer

model = VAETrainer.load_from_checkpoint("checkpoints/epoch=0-step=15620-v1.ckpt")

# export for turbulence removal project
torch.save(model.vae.decoder.state_dict(), "/home/hudson/code/turbulence_model/models/turb_vae/decoder.pt")


torch.set_grad_enabled(False)

# test the reconstructions

dataset = VonKarmanXY(
    num_samples=30,
    resolution=(24, 24),
    x_range=(-1, 1),
    y_range=(-1, 1),
    L0_vals=np.logspace(-1, 1, 30),
    L0_probs=None,
    vk_cache=True,
    base_seed=0,
    tfms=nn.Upsample(size=(64, 64), mode="bicubic"),
)

# test reconstructions
for ix, (n, L0) in enumerate(dataset):
    n = n[None]
    n_hat, _, _ = model.vae(n.to("cuda"), sample=False)

    # plot n and n_hat side by side
    n = n[0, 0].cpu().numpy()
    n_hat = n_hat[0, 0].cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    im1 = axs[0].imshow(n, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(n_hat, cmap="gray")
    axs[1].set_title("Reconstructed")
    fig.tight_layout()
    plt.savefig(f"plots/reconstructions/L0={L0:0.2f}_{ix}.png")

# test random generation
z = torch.randn(30, model.vae.decoder.in_channels, 8, 8).to("cuda")
n_hat = model.vae.decoder(z).cpu().numpy().squeeze()

fig, axs = plt.subplots(5, 6, figsize=(12, 10))
for ix, n in enumerate(n_hat):
    row = ix // 6
    col = ix % 6
    axs[row, col].imshow(n, cmap="gray")
    axs[row, col].axis("off")
plt.tight_layout()
plt.savefig("plots/generation/all_n_hat.png")

