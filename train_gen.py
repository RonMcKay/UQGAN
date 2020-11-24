from logging import Logger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred import Experiment
from sacred.config.config_dict import ConfigDict
from sacred.run import Run
from torch.utils.data import DataLoader

from cls_models import cls_models, load_cls_model
from cls_models.base import BaseClassifier
from config import Config
from datasets import datasets, load_data
from gan_models import gan_models, load_gan_model
from gen import GEN
from logging_utils import log_config
from logging_utils.lightning_sacred import SacredLogger
from options import print_options
from uqgan import CustomCheckpointIO
from utils import (
    TimeEstimator,
    get_accelerator_device,
    init_experiment,
    register_exp_folder,
)
from vae_models import load_vae_model, vae_models

ex = Experiment("train_gen", ingredients=[datasets, gan_models, cls_models, vae_models])
init_experiment(ex)
sacred_logger = SacredLogger(ex)


@cls_models.config
def cls_models_config_update(cfg):
    cfg["method"] = "gen"


@datasets.config
def dataset_config_update(cfg):
    cfg["static"] = False
    cfg["mode"] = "train"


@vae_models.config
def vae_config_update(cfg, dataset):
    cfg["latent_dim"] = 50


@gan_models.config
def gan_config_update(cfg, dataset):
    cfg["name"] = "sensoyetal2020"
    cfg["latent_dim"] = 50
    cfg.pop("disc_checkpoint", None)
    cfg["disc_latent_checkpoint"] = None
    cfg["disc_img_checkpoint"] = None


@ex.config
def config(dataset):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        epochs=100,
        vae_iterations=10,
        batch_size=256,
        gpu=0,
        save_folder=Config.root_save_folder,
        num_workers=8,
        ood_datasets=None,
    )

    opt = dict(  # noqa: F841
        lr=1e-4,
        lr_cls=1e-3,
        lr_vae=1e-3,
        weight_decay=2e-4,
        weight_decay_cls=1e-3,
    )


@ex.command(unobserved=True)
def options(args, opt, dataset):
    used_options = set(
        [
            "enable_progress_bar",
            "check_val_every_n_epoch",
            "min_lr_vae",
            "lr_disc_image",
            "lr_disc_latent",
            "lr_gen",
            "min_lr_disc_image",
            "min_lr_disc_latent",
            "min_lr_gen",
            "weight_decay_disc_image",
            "weight_decay_disc_latent",
            "weight_decay_gen",
            "cls_models",
            "datasets",
        ]
    )
    used_options = used_options.union(
        set(list(args.keys()) + list(opt.keys()) + list(dataset["cfg"].keys()))
    )

    print_options(used_options)


@ex.automain
def main(  # type: ignore
    args: ConfigDict,
    opt: ConfigDict,
    gan_model: ConfigDict,
    cls_model: ConfigDict,
    vae_model: ConfigDict,
    dataset: ConfigDict,
    _run: Run,
    _log: Logger,
):
    log_config(_run, _log)
    exp_folder = register_exp_folder(args["save_folder"], _run)

    ########################################
    #              Set devices
    ########################################

    accelerator, devices = get_accelerator_device(args["gpu"])

    ########################################
    #       Load dataset and model
    ########################################
    traindat, sampler = load_data()
    valdat, _ = load_data(static=True, mode="eval")

    trainloader = DataLoader(
        traindat,
        batch_size=args["batch_size"],
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=args["num_workers"],
    )

    valloader = DataLoader(
        valdat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    classifier = load_cls_model(cl_dim=dataset["cfg"]["cl_dim"])  # type: BaseClassifier
    generator, disc_latent, disc_img = load_gan_model()
    vae = load_vae_model()

    gen = GEN(
        classifier=classifier,
        generator=generator,
        discriminator_image=disc_img,
        discriminator_latent=disc_latent,
        vae=vae,
        args=args,
        dataset=dataset,
        opt=opt,
    )

    checkpoint_callback = ModelCheckpoint(
        exp_folder,
        monitor="val_acc",
        mode="max",
        save_last=True,
        filename="gen",
    )

    time_estimator_callback = TimeEstimator(args["epochs"], logger=_log)

    custom_checkpoint_io = CustomCheckpointIO(
        modules=(
            "classifier",
            "generator",
            "discriminator_latent",
            "discriminator_image",
            "vae",
        )
    )

    trainer = Trainer(
        default_root_dir=exp_folder,
        logger=sacred_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, time_estimator_callback],
        plugins=[custom_checkpoint_io],
        max_epochs=args["epochs"],
        enable_progress_bar=args.get("enable_progress_bar", False),
        log_every_n_steps=5,
        check_val_every_n_epoch=args.get("check_val_every_n_epoch", 1),
    )

    ########################################
    #            Training
    ########################################

    trainer.fit(gen, train_dataloaders=trainloader, val_dataloaders=valloader)
