from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from turb_vae.data_generation.dataset import VonKarmanXY
from turb_vae.vae.layers import Decoder2d, Encoder2d


class TurbVaeConfig:
    # model
    vae_config = dict(
        encoder = Encoder2d(1, 4, (1, 3, 3, 3), (64,) * 4, "relu"),
        decoder = Decoder2d(4, 1, (3, 3, 3, 1), (64,) * 4, 2, "relu"),
        kl_weight = 0.01,
        learning_rate = 1e-4
    )

    # dataset
    train_dataset = VonKarmanXY(
        num_samples=int(1e6),
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
    val_dataset.num_samples = int(1e4)
    test_dataset = deepcopy(train_dataset)
    test_dataset.base_seed = 12345
    test_dataset.num_samples = int(1e4)

    # dataloaders
    num_workers = 15
    fit_params = dict(
        train_dataloaders = DataLoader(
            train_dataset, batch_size=64, num_workers=num_workers, persistent_workers=True
        ),
        val_dataloaders = DataLoader(
            val_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )
    test_params = dict(
        test_dataloaders = DataLoader(
            test_dataset, batch_size=int(1e3), num_workers=num_workers, persistent_workers=True
        )
    )

    # training
    logger = WandbLogger(project="turb_vae", name="vae_train", offline=False)
    checkpoint_callback = ModelCheckpoint(
        "checkpoints/", save_top_k=1, monitor="val_loss", 
    )
    trainer = pl.Trainer(
        precision="16-mixed",
        log_every_n_steps=100,
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
    )