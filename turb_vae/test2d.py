import matplotlib.pyplot as plt
import torch
from config import ProjConfig as cfg
from train2d import VAETrainer

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAETrainer.load_from_checkpoint("checkpoints/epoch=0-step=125000-v1.ckpt").to(device)

# export for turbulence removal project
# torch.save(model.vae.decoder.state_dict(), "/home/hudson/code/turbulence_model/models/turb_vae/decoder_kl_1.pt")

# test the reconstructions
dataset = cfg.test_dataset
dataset.num_samples = 10

# test reconstructions
for ix, (n, L0) in enumerate(dataset):
    n = n[None]
    n_hat, dist = model.vae(n.to(device))

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
    plt.savefig(f"plots/reconstructions/L0={L0:0.2f}_{ix}.png")

# test random generation
z = torch.randn(30, model.vae.decoder.in_channels, 8, 8).to(device)
n_hat = model.vae.decoder(z).cpu().numpy().squeeze()

fig, axs = plt.subplots(5, 6, figsize=(12, 10))
for ix, n in enumerate(n_hat):
    row = ix // 6
    col = ix % 6
    axs[row, col].imshow(n, cmap="gray")
    axs[row, col].axis("off")
plt.tight_layout()
plt.savefig("plots/all_n_hat.png")

