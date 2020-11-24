from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred import Experiment
from torch.utils.data import DataLoader

from cls_models import cls_models, load_cls_model
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

ex = Experiment("train_classifier", ingredients=(cls_models, datasets))
init_experiment(ex)

sacred_logger = SacredLogger(ex)


@ex.config
def config(dataset, cls_model):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        epochs=200,
        batch_size=256,
        gpu=0,
        save_folder=Config.root_save_folder,
        num_workers=8,
    )

    if cls_model["cfg"].get("mc_dropout", 0.0) > 0:
        args["mc_samples"] = 5

    opt = dict(  # noqa: F841
        lr=1e-3,
        min_lr=1e-5,
        weight_decay=2e-4,
    )


@ex.command(unobserved=True)
def options(args, opt, dataset, cls_model):
    used_options = set(["enable_progress_bar", "mc_samples", "datasets", "cls_models"])
    used_options = used_options.union(
        set(
            list(args.keys())
            + list(opt.keys())
            + list(dataset["cfg"].keys())
            + list(cls_model["cfg"].keys())
        )
    )

    print_options(used_options)


@ex.automain
def main(args, opt, cls_model, dataset, _run, _log):
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

    trainloader = DataLoader(
        traindat,
        batch_size=args["batch_size"],
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=args["num_workers"],
    )

    valdat, _ = load_data(mode="eval", static=True)
    valloader = DataLoader(
        valdat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    classifier = load_cls_model(cl_dim=dataset["cfg"]["cl_dim"], opt=opt)

    if classifier.method == "uqgan":
        classifier.compute_rel_class_frequencies(trainloader)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_folder,
        monitor="val_acc",
        mode="max",
        filename="classifier",
        save_last=True,
    )

    time_estimator_callback = TimeEstimator(max_iterations=args["epochs"], logger=_log)

    trainer = Trainer(
        default_root_dir=exp_folder,
        logger=sacred_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, time_estimator_callback],
        max_epochs=args["epochs"],
        enable_progress_bar=args.get("enable_progress_bar", False),
    )

    ########################################
    #            Training
    ########################################

    trainer.fit(classifier, train_dataloaders=trainloader, val_dataloaders=valloader)
