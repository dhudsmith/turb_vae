import inspect
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl

# from turb_vae.data_generation.dataset import VonKarmanXY
# from turb_vae.vae.layers import Decoder2d, Encoder2d
from data_generation.dataset import VonKarmanXY
from kl_scheduler import KLScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from vae.layers import Decoder2d, Encoder2d
from vae.vae import LowRankVariationalAutoencoder


class BaseConfig:
    @classmethod
    def to_dict(cls):
        return {k: repr(v) for k, v in cls.__dict__.items() if not k.startswith("__") and not inspect.ismethod(v) and k!= "to_dict"}

class TurbVaeConfig(BaseConfig):
    # model
    vae_config = dict(
        rank = (r:=10),
        encoder = Encoder2d(1, (c1:=16)*(2+r), (1, 3, 3, 3), (c2:=64,) * 4, "relu"),
        decoder = Decoder2d(c1, 1, (3, 3, 3, 1), (c2,) * 4, "relu"),
        num_particles = 3,
        cov_factor_init_scale = 1e-3,    
    )
    del r, c1, c2

    kl_scheduler = KLScheduler(1e-3, 1e-1, int(9e4))
    learning_rate = 1e-4

    # dataset
    train_dataset = VonKarmanXY(
        num_samples=int(1e7),
        resolution=(24, 24),
        x_range=(-1, 1),
        y_range=(-1, 1),
        L0_vals=np.logspace(-1, 1, 30),
        L0_probs=None,
        vk_cache=True,
        base_seed=0,
        tfms=nn.Upsample(size=(64, 64), mode="bicubic"),
    )
    val_dataset = deepcopy(train_dataset)
    val_dataset.base_seed = 777
    val_dataset.num_samples = int(1e3)
    test_dataset = deepcopy(train_dataset)
    test_dataset.base_seed = 12345
    test_dataset.num_samples = int(1e3)

    # dataloaders
    num_workers = 19
    fit_params = dict(
        train_dataloaders = DataLoader(
            train_dataset, batch_size=64, num_workers=num_workers, persistent_workers=True
        ),
        val_dataloaders = DataLoader(
            val_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )
    test_params = dict(
        dataloaders = DataLoader(
            test_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )

    # training
    logger = WandbLogger(project="turb_vae", offline=False)
    checkpoint_callback = ModelCheckpoint(
        "checkpoints/", save_top_k=1, monitor="val_loss", 
    )
    trainer = pl.Trainer(
        log_every_n_steps=100,
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        gradient_clip_val=2.0,
    )

class ProjConfig(BaseConfig):
    # model
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

    kl_scheduler = KLScheduler(1e-4, 1, int(9e6))
    learning_rate = 1e-4

    # dataset
    train_dataset = VonKarmanXY(
        num_samples=int(1e7),
        resolution=(24, 24),
        x_range=(-1, 1),
        y_range=(-1, 1),
        L0_vals=np.logspace(-1, 1, 30),
        L0_probs=None,
        vk_cache=True,
        base_seed=0,
        tfms=nn.Upsample(size=vae_config['input_size'], mode="bicubic"),
    )
    val_dataset = deepcopy(train_dataset)
    val_dataset.base_seed = 777
    val_dataset.num_samples = int(1e3)
    test_dataset = deepcopy(train_dataset)
    test_dataset.base_seed = 12345
    test_dataset.num_samples = int(1e3)

    # dataloaders
    num_workers = 40
    fit_params = dict(
        train_dataloaders = DataLoader(
            train_dataset, batch_size=64, num_workers=num_workers, persistent_workers=True
        ),
        val_dataloaders = DataLoader(
            val_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )
    test_params = dict(
        dataloaders = DataLoader(
            test_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )

    # training
    logger = WandbLogger(project="turb_vae", offline=False)
    checkpoint_callback = ModelCheckpoint(
        "checkpoints/", save_top_k=1, monitor="val_elbo", 
    )
    trainer = pl.Trainer(
        log_every_n_steps=100,
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        gradient_clip_val=2.0,
    )
    
if __name__ == "__main__":
    import json
    cfg = ProjConfig()

    print(json.dumps(cfg.to_dict(), indent=2))

    vae = LowRankVariationalAutoencoder(**cfg.vae_config)

    def num_pars(module):
        return sum(p.numel() for p in module.parameters())/1e6
    
    print(f"VAE num parameters: {num_pars(vae):.2f}M")
    print(f"Encoder num parameters: {num_pars(vae.encoder):.2f}M")
    print(f"Decoder num parameters: {num_pars(vae.decoder):.2f}M")
    print(f"fc_encode num parameters: {num_pars(vae.fc_encode):.2f}M")
    print(f"fc_decode num parameters: {num_pars(vae.fc_decode):.2f}M")
    