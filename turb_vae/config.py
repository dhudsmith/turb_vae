

class ClassDictMixin:
    @classmethod
    def to_dict(cls):
        # get all of the class (not instance) attributes as a dictionary

        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}


class LightningModelConfig(ClassDictMixin):
    vae_config = dict(
        encoder_kwargs=dict(
            in_channels=2,
            out_channels=8,
            num_blocks=(1, 3, 3, 3),
            block_out_channels=(64,) * 4,
            act_fn="relu",
        ),
        decoder_kwargs=dict(
            in_channels=4,
            out_channels=1,
            num_blocks=(3, 3, 3, 1),
            block_out_channels=(64,) * 4,
            act_fn="relu",
        ),
        rank=0,
        is_conditional=True,
        embed_dim=512,
        input_size=(64, 64),
        num_particles=5,
        cov_factor_init_scale=1.0,
    )

    kl_scheduler_kwargs = dict(start=1e-7, stop=1, n_samples=int(9e5))

    learning_rate = 1e-4


class DataConfig(ClassDictMixin):
    upsample_kwargs = dict(
        size=LightningModelConfig.vae_config["input_size"], mode="bicubic"
    )

    L0_vals_logspace_kwargs = dict(start=-1, stop=1, num=30)

    train_dataset_kwargs = dict(
        num_samples=int(1e6),
        resolution=(8, 8),
        x_range=(-1, 1),
        y_range=(-1, 1),
        L0_probs=None,
        vk_cache=True,
        base_seed=0,
    )

    val_dataset_kwargs = train_dataset_kwargs.copy()
    val_dataset_kwargs["num_samples"] = int(1e4)
    val_dataset_kwargs["base_seed"] = 777

    test_dataset_kwargs = train_dataset_kwargs.copy()
    test_dataset_kwargs["num_samples"] = int(1e4)
    test_dataset_kwargs["base_seed"] = 12345

    # dataloaders
    train_dataloader_kwargs = dict(
        batch_size=64, num_workers=15, persistent_workers=True
    )
    val_dataloader_kwargs = train_dataloader_kwargs.copy()
    val_dataloader_kwargs["batch_size"] = int(1e3)
    test_dataloader_kwargs = val_dataloader_kwargs.copy()


class TrainingConfig(ClassDictMixin):
    logger_kwargs = dict(project="turb_vae", offline=True)
    checkpoint_kwargs = dict(dirpath="checkpoints/", save_top_k=1, monitor="val_elbo")
    trainer_kwargs = dict(
        log_every_n_steps=100,
        max_epochs=1,
        val_check_interval=0.01,
        gradient_clip_val=5.0,
    )


if __name__ == "__main__":
    import json

    from vae.vae import VAE

    model_cfg = LightningModelConfig()
    data_cfg = DataConfig()
    training_cfg = TrainingConfig()

    for cfg in [model_cfg, data_cfg, training_cfg]:
        print(cfg)
        print(json.dumps(cfg.to_dict(), indent=4))

    # test number of parameters
    vae = VAE(**LightningModelConfig.vae_config)  # type: ignore

    def num_pars(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    print(f"VAE num parameters: {num_pars(vae):.2f}M")
    print(f"Encoder num parameters: {num_pars(vae.encoder):.2f}M")
    print(f"Decoder num parameters: {num_pars(vae.decoder):.2f}M")
    print(f"fc_encode num parameters: {num_pars(vae.fc_encode):.2f}M")
    print(f"fc_decode num parameters: {num_pars(vae.fc_decode):.2f}M")
