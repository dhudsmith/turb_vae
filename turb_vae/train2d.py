import pytorch_lightning as pl
import torch

from turb_vae.vae.layers import Decoder2d, Encoder2d
from turb_vae.vae.vae import LowRankMultivariateNormal, LowRankVariationalAutoencoder

# from vae.layers import Decoder2d, Encoder2d
# from vae.vae import LowRankMultivariateNormal, LowRankVariationalAutoencoder

torch.set_float32_matmul_precision("medium")


class VAETrainer(pl.LightningModule):
    def __init__(self, encoder: Encoder2d, decoder: Decoder2d, rank = 3, num_particles = 1, kl_weight: float = 1.0, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.vae = LowRankVariationalAutoencoder(
            encoder,
            decoder,
            rank = rank,
            num_particles = num_particles
        )

        self.kl_weight: float = kl_weight
        self.learning_rate: float = learning_rate

    def training_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def test_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("test_loss", loss)
        self.log("test_kl_loss", kl_loss)
        self.log("test_recon_loss", recon_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def get_losses(self, batch):
        n, _ = batch
        n_hat: torch.FloatTensor
        dist: LowRankMultivariateNormal
        n_hat, dist = self.vae(n)

        recon_loss = torch.nn.functional.mse_loss(n_hat, n)
        kl_loss = dist.kl_divergence().mean()
        loss =  recon_loss + self.kl_weight * kl_loss

        return loss, recon_loss, kl_loss


if __name__ == "__main__":
    from config import TurbVaeConfig as cfg
    model = VAETrainer(**cfg.vae_config).to("cuda")
    trainer = cfg.trainer
    trainer.fit(model, **cfg.fit_params)
    trainer.test(model, **cfg.test_params)
