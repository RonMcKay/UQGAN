import matplotlib
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred import Experiment
from torch.utils.data import DataLoader

matplotlib.use("Agg")

from cae_models import cae_models, load_cae_model
from config import Config
from datasets import datasets, load_data
from logging_utils import log_config
from logging_utils.lightning_sacred import SacredLogger
from options import print_options
from utils import (
    TimeEstimator,
    get_accelerator_device,
    init_experiment,
    register_exp_folder,
)

ex = Experiment("train_cae", ingredients=[datasets, cae_models])
init_experiment(ex)

sacred_logger = SacredLogger(ex)


@ex.config
def config(dataset):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        epochs=200,
        batch_size=256,
        gpu=0,
        save_folder=Config.root_save_folder,
        num_workers=12,
        num_sample_images=4,
        sample_every_n=5,
    )

    opt = dict(  # noqa: F841
        lr=1e-3,
        weight_decay=0,
    )


@ex.command(unobserved=True)
def options(args, opt, dataset, cae_model):
    used_options = set(["enable_progress_bar", "cae_models", "datasets"])
    used_options = used_options.union(
        set(
            list(args.keys())
            + list(opt.keys())
            + list(dataset["cfg"].keys())
            + list(cae_model["cfg"].keys())
        )
    )

    print_options(used_options)


@ex.automain
def main(args, opt, _run, _log):
    log_config(_run, _log)
    exp_folder = register_exp_folder(args["save_folder"], _run)

    ########################################
    #       Load dataset and model
    ########################################
    traindat, sampler = load_data()
    valdat, _ = load_data(mode="eval", static=True)

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

    cae = load_cae_model(
        num_sample_images=args["num_sample_images"],
        sample_every_n=args["sample_every_n"],
        opt=opt,
    )

    checkpoint_callback = ModelCheckpoint(
        exp_folder,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="cae",
    )

    time_estimator_callback = TimeEstimator(
        max_iterations=args["epochs"],
        logger=_log,
    )

    accelerator, devices = get_accelerator_device(args["gpu"])

    trainer = Trainer(
        default_root_dir=exp_folder,
        logger=sacred_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, time_estimator_callback],
        max_epochs=args["epochs"],
        enable_progress_bar=False,
    )

    ############################
    #       Training
    ############################

    trainer.fit(cae, train_dataloaders=trainloader, val_dataloaders=valloader)
