import pytorch_lightning as pl
import torch
from config import BaseConfig
from pytorch_lightning.loggers import WandbLogger

# from turb_vae.vae.layers import Decoder2d, Encoder2d
# from turb_vae.vae.vae import LowRankMultivariateNormal, LowRankVariationalAutoencoder
from vae.vae import LowRankMultivariateNormal, LowRankVAE

torch.set_float32_matmul_precision("medium")


class VAETrainer(pl.LightningModule):
    def __init__(self, cfg: BaseConfig): 
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        
        self.vae = LowRankVAE(
            **cfg.vae_config
        )

        self.kl_scheduler = cfg.kl_scheduler
        self.learning_rate: float = cfg.learning_rate

    def training_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True, on_step=True)

        # step the kl loss weight
        num_samples = batch[0].shape[0]
        self.kl_scheduler.step(num_samples)
        self.log("kl_weight", self.kl_scheduler.get_kl_weight(), prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "val_recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("val_elbo", recon_loss + kl_loss, prog_bar=True, on_step=False, on_epoch=True)

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

        # we unsqueeze n to add the dummy particle dimension
        normalization = n.shape[0] * n.shape[1] * n.shape[2] * n.shape[3]  # the total number of pixels in the batch
        recon_loss = torch.nn.functional.mse_loss(n_hat, n.unsqueeze(0), reduction="sum") / self.vae.num_particles / normalization  # average the loss by the number of particles
        kl_loss = dist.kl_divergence().sum() / normalization
        beta = self.kl_scheduler.get_kl_weight()
        loss =  recon_loss + beta * kl_loss

        return loss, recon_loss, kl_loss
    
    def on_train_start(self):
        # save hyperparameters
        logger: WandbLogger = self.logger
        logger.experiment.config.update(cfg.to_dict()) 

if __name__ == "__main__":
    from config import ProjConfig as cfg
    model = VAETrainer(cfg).to("cuda")
    trainer = cfg.trainer
    trainer.fit(model, **cfg.fit_params)
    trainer.test(model, **cfg.test_params)
