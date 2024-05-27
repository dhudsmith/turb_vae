import numpy as np
import pytorch_lightning as pl
import torch
from data_generation.dataset import VonKarmanXY
from kl_scheduler import KLScheduler
from torch import nn
from torch.utils.data import DataLoader

# from turb_vae.vae.layers import Decoder2d, Encoder2d
# from turb_vae.vae.vae import LowRankMultivariateNormal, LowRankVariationalAutoencoder
from vae.vae import LowRankMultivariateNormal, LowRankVAE

torch.set_float32_matmul_precision("medium")


class VAETrainer(pl.LightningModule):
    def __init__(
        self, vae_kwargs: dict, kl_scheduler_kwargs: dict, learning_rate: float
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vae = LowRankVAE(**vae_kwargs)
        self.kl_scheduler = KLScheduler(**kl_scheduler_kwargs)
        self.learning_rate: float = learning_rate

    def training_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True, on_step=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True, on_step=True)

        # step the kl loss weight
        num_samples = batch[0].shape[0]
        self.kl_scheduler.step(num_samples)
        self.log(
            "kl_weight",
            self.kl_scheduler.get_kl_weight(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, _):  # type: ignore
        loss, recon_loss, kl_loss = self.get_losses(batch)

        # log losses
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "val_recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_elbo",
            recon_loss + kl_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
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
        n, L0 = batch
        if len(L0.shape) == 1:
            L0 = L0[:, None]

        n_hat: torch.FloatTensor
        dist: LowRankMultivariateNormal
        n_hat, dist = self.vae(n, L0)

        # we unsqueeze n to add the dummy particle dimension
        normalization = (
            n.shape[0] * n.shape[1] * n.shape[2] * n.shape[3]
        )  # the total number of pixels in the batch
        recon_loss = (
            torch.nn.functional.mse_loss(n_hat, n.unsqueeze(0), reduction="sum")
            / self.vae.num_particles
            / normalization
        )  # average the loss by the number of particles
        kl_loss = dist.kl_divergence().sum() / normalization
        beta = self.kl_scheduler.get_kl_weight()
        loss = recon_loss + beta * kl_loss

        return loss, recon_loss, kl_loss


if __name__ == "__main__":
    from config import DataConfig as data_cfg
    from config import LightningModelConfig as model_cfg
    from config import TrainingConfig as train_cfg
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    model = VAETrainer(
        vae_kwargs=model_cfg.vae_config,
        kl_scheduler_kwargs=model_cfg.kl_scheduler_kwargs,
        learning_rate=model_cfg.learning_rate,
    )

    # datasets
    upsample = nn.Upsample(**data_cfg.upsample_kwargs)  # type: ignore
    L0_vals = np.logspace(**data_cfg.L0_vals_logspace_kwargs)  # type: ignore
    train_dataset = VonKarmanXY(
        **data_cfg.train_dataset_kwargs,  # type: ignore
        L0_vals=L0_vals,
        tfms=upsample,
    )
    val_dataset = VonKarmanXY(
        **data_cfg.val_dataset_kwargs,  # type: ignore
        L0_vals=L0_vals,
        tfms=upsample,
    )
    test_dataset = VonKarmanXY(
        **data_cfg.test_dataset_kwargs,  # type: ignore
        L0_vals=L0_vals,
        tfms=upsample,
    )

    # dataloaders
    train_dataloader = DataLoader(train_dataset, **data_cfg.train_dataloader_kwargs)  # type: ignore
    val_dataloader = DataLoader(val_dataset, **data_cfg.val_dataloader_kwargs)  # type: ignore
    test_dataloader = DataLoader(test_dataset, **data_cfg.test_dataloader_kwargs)  # type: ignore

    # logger
    logger = WandbLogger(**train_cfg.logger_kwargs)  # type: ignore
    logger.experiment.config.update(model_cfg.to_dict())
    logger.experiment.config.update(data_cfg.to_dict())
    logger.experiment.config.update(train_cfg.to_dict())

    # training
    checkpoint = ModelCheckpoint(**train_cfg.checkpoint_kwargs)  # type: ignore
    trainer = pl.Trainer(
        **train_cfg.trainer_kwargs,  # type: ignore
        logger=logger,
        callbacks=[checkpoint],
    )

    # fit and test
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model, dataloaders=test_dataloader)
