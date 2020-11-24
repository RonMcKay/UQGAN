from logging import Logger

import matplotlib
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred import Experiment
from sacred.config.config_dict import ConfigDict
from sacred.run import Run
from torch.utils.data import DataLoader

matplotlib.use("Agg")

from cls_models import cls_models, load_cls_model
from confident_classifier import ConfidentClassifier
from config import Config
from datasets import datasets, load_data
from gan_models import gan_models, load_gan_model
from logging_utils import log_config
from logging_utils.lightning_sacred import SacredLogger
from uqgan import CustomCheckpointIO
from utils import (
    TimeEstimator,
    get_accelerator_device,
    init_experiment,
    register_exp_folder,
)

ex = Experiment(
    "train_confident_classifier", ingredients=[datasets, gan_models, cls_models]
)
init_experiment(ex)
sacred_logger = SacredLogger(ex)


@datasets.config
def dataset_config_update(cfg):
    cfg["static"] = False
    cfg["mode"] = "train"


@gan_models.config
def gan_config_update(cfg):
    cfg["name"] = "dcgan"
    cfg["latent_dim"] = 100
    cfg["conditional"] = False


@cls_models.config
def cls_config_update(cfg):
    cfg["method"] = "softmax"


@ex.config
def config(dataset):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        epochs=100,  # Total generator iterations
        batch_size=256,
        gpu=0,
        beta=1.0,
        save_folder=Config.root_save_folder,
        num_workers=8,
        ood_datasets=None,
    )

    opt = dict(  # noqa: F841
        lr=2e-4,
        min_lr=1e-5,
        weight_decay=2e-4,
    )


@ex.automain
def main(  # type: ignore
    args: ConfigDict,
    opt: ConfigDict,
    gan_model: ConfigDict,
    cls_model: ConfigDict,
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
    #       Load datasets and models
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

    classifier = load_cls_model(cl_dim=dataset["cfg"]["cl_dim"])

    generator, discriminator = load_gan_model()

    confident_classifier = ConfidentClassifier(
        classifier=classifier,
        generator=generator,
        discriminator=discriminator,
        args=args,
        dataset=dataset,
        opt=opt,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_folder,
        monitor="val_acc",
        mode="max",
        filename="confidentclassifier",
        save_last=True,
    )
    time_estimator_callback = TimeEstimator(args["epochs"], logger=_log)

    custom_checkpoint_io = CustomCheckpointIO()

    trainer = Trainer(
        default_root_dir=exp_folder,
        logger=sacred_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, time_estimator_callback],
        plugins=[custom_checkpoint_io],
        max_epochs=args["epochs"],
        enable_progress_bar=False,
    )

    ########################################
    #            Training
    ########################################

    trainer.fit(
        confident_classifier, train_dataloaders=trainloader, val_dataloaders=valloader
    )
