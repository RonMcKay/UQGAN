from functools import partial

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred import Experiment
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

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

ex = Experiment("train_meta_classifier", ingredients=(cls_models, datasets))
init_experiment(ex)
sacred_logger = SacredLogger(ex)


def target_transformation(target, hint):
    return [target, hint]


@cls_models.config
def cls_models_config_update(cfg):
    cfg["meta_classifier"] = True


@ex.config
def config(dataset):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        epochs=100,
        batch_size=256,
        gpu=0,
        save_folder=Config.root_save_folder,
        num_workers=8,
        ood_datasets=None,
    )

    opt = dict(  # noqa: F841
        lr=2e-4,
        min_lr=1e-5,
        weight_decay=2e-4,
    )


@ex.command(unobserved=True)
def options(args, opt, dataset, cls_model):
    used_options = set(["enable_progress_bar", "datasets", "cls_models"])
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
    #       Load datasets and models
    ########################################

    indat, _ = load_data(target_transform=partial(target_transformation, hint=1))

    invaldat, _ = load_data(
        target_transform=partial(target_transformation, hint=1), mode="eval"
    )

    ood_datasets = []
    ood_val_datasets = []

    if args["ood_datasets"] is None:
        raise ValueError("No OOD dataset specified")
    else:
        for d in args["ood_datasets"].split(","):
            ood_cfg = dict(
                name=d,
                mode=dataset["cfg"]["mode"],
                static=True,
                image_channels=dataset["cfg"]["image_channels"],
                target_transform=partial(target_transformation, hint=0),
            )
            if "image_size" in ood_cfg:
                ood_cfg["image_size"] = dataset["cfg"]["image_size"]
            ood_datasets.append(load_data(**ood_cfg)[0])

            ood_cfg["mode"] = {"train": "eval", "eval": "train"}.get(
                dataset["cfg"]["mode"], "eval"
            )
            ood_val_datasets.append(load_data(**ood_cfg)[0])

    ooddat = ConcatDataset(ood_datasets)
    ood_val_dat = ConcatDataset(ood_val_datasets)

    traindat = ConcatDataset([indat, ooddat])
    valdat = ConcatDataset([invaldat, ood_val_dat])

    # sample weights for an equal amount of samples from all provided OOD datasets
    ood_weights = []
    for ood_dataset in ood_datasets:
        ood_weights += [1.0 / (len(ood_dataset) * len(ood_datasets))] * len(ood_dataset)

    # Initialize the sampler so that we have an equal amount of
    # in-/out-of-distribution samples
    sampler = WeightedRandomSampler(
        [1.0 / len(indat)] * len(indat) + ood_weights, num_samples=len(indat) * 2
    )

    trainloader = DataLoader(
        traindat,
        batch_size=args["batch_size"],
        shuffle=False,
        sampler=sampler,
        num_workers=args["num_workers"],
    )

    valloader = DataLoader(
        valdat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    classifier = load_cls_model(opt=opt)
    if classifier.method == "uqgan":
        classifier.compute_rel_class_frequencies(
            DataLoader(
                indat,
                batch_size=args["batch_size"],
                shuffle=False,
                num_workers=args["num_workers"],
            )
        )

    time_estimator_callback = TimeEstimator(max_iterations=args["epochs"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_folder,
        monitor="val_acc",
        mode="max",
        filename="classifier",
        save_last=True,
    )

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
