from logging import Logger

import matplotlib
from pytorch_lightning import Trainer
from sacred import Experiment
from sacred.config.config_dict import ConfigDict
from sacred.run import Run
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from copy import deepcopy

from pytorch_lightning.callbacks import ModelCheckpoint

from cae_models import cae_models, load_cae_model
from cae_models.identity import IdentityCAE
from cls_models import cls_models, load_cls_model
from cls_models.base import BaseClassifier
from config import Config
from datasets import datasets, load_data
from gan_models import gan_models, load_gan_model
from logging_utils import log_config
from logging_utils.lightning_sacred import SacredLogger
from options import print_options
from uqgan import UQGAN, CustomCheckpointIO
from utils import (
    AVAILABLE_REG_TYPES,
    TimeEstimator,
    get_accelerator_device,
    init_experiment,
    register_exp_folder,
)

ex = Experiment(
    "train_uqgan",
    ingredients=[datasets, gan_models, cls_models, cae_models],
)
init_experiment(ex)

sacred_logger = SacredLogger(ex)

xymin = None
xymax = None


@cls_models.config
def cls_models_config_update(cfg):
    cfg["method"] = "uqgan"


@datasets.config
def dataset_config_update(cfg):
    cfg["static"] = False
    cfg["mode"] = "train"


@ex.config
def config(dataset, cls_model):
    tags = [dataset["cfg"]["name"]]  # noqa: F841

    args = dict(  # noqa: F841
        iterations=10000,  # Total generator iterations
        batch_size=256,
        discriminator_iterations=5,
        classifier_iterations=5,
        gpu=0,
        lambda_gp=10,
        lambda_reg_loss=32,
        lambda_cl_loss=2,
        lambda_real_ood=0.6,
        reg_type="logcosine",
        save_folder=Config.root_save_folder,
        num_workers=8,
        val_check_interval=2,
        ood_datasets=None,  # datasets to use for evaluating ood detection performance
    )

    opt = dict(  # noqa: F841
        lr=2e-4,
        lr_cls=1e-3,
        min_lr=1e-5,
        weight_decay=0,
    )

    if cls_model["cfg"].get("mc_dropout", 0.0) > 0:
        args["mc_samples"] = 5

    if "reg_weight" in args:
        if len(args["reg_type"].split(",")) != len(
            args["reg_weight"]
        ) or not isinstance(  # type: ignore
            args["reg_weight"], list
        ):
            raise ValueError(
                (
                    "Invalid 'reg_weight' config. "
                    "'reg_weight' should be a comma separated list of numers "
                    "with the same length as 'reg_type'. "
                    f"Got 'reg_type': {args['reg_type']} and "
                    f"'reg_weight': {args['reg_weight']}"
                )
            )

    for reg_t in args["reg_type"].split(","):  # type: ignore
        if reg_t not in AVAILABLE_REG_TYPES:
            raise ValueError(f"Unknown regularization type '{reg_t}'")
    del reg_t


@ex.command(unobserved=True)
def options(args, opt, dataset, cls_model, cae_model):
    used_options = set(
        [
            "enable_progress_bar",
            "lr_disc",
            "lr_gen",
            "mc_samples",
            "min_lr_cls",
            "min_lr_disc",
            "min_lr_gen",
            "weight_decay_cls",
            "weight_decay_disc",
            "weight_decay_gen",
            "cls_models",
            "datasets",
        ]
    )
    used_options = used_options.union(
        set(
            list(args.keys())
            + list(opt.keys())
            + list(dataset["cfg"].keys())
            + list(cls_model["cfg"].keys())
            + list(cae_model["cfg"].keys())
        )
    )

    print_options(used_options)


@ex.automain
def main(  # type: ignore
    args: ConfigDict,
    opt: ConfigDict,
    gan_model: ConfigDict,
    cae_model: ConfigDict,
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
        drop_last=True,
    )

    valloader = DataLoader(
        valdat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    classifier = load_cls_model(cl_dim=dataset["cfg"]["cl_dim"])  # type: BaseClassifier
    classifier.compute_rel_class_frequencies(trainloader)

    if dataset["cfg"]["name"].lower().startswith("toy"):
        overwrite_gan_cfg = dict(
            name="toy", output_size=2, input_size=2, cl_dim=traindat.CL_DIM
        )
    else:
        overwrite_gan_cfg = dict(
            name="toy",
            output_size=cae_model["cfg"]["latent_dim"],
            input_size=cae_model["cfg"]["latent_dim"],
            cl_dim=dataset["cfg"]["cl_dim"],
        )

    generator, discriminator = load_gan_model(**overwrite_gan_cfg)

    gen_config = deepcopy(gan_model)
    gen_config.update(overwrite_gan_cfg)

    if dataset["cfg"]["name"].lower().startswith("toy"):
        cae = IdentityCAE()
    else:
        cae = load_cae_model()

    uqgan = UQGAN(
        classifier=classifier,
        generator=generator,
        discriminator=discriminator,
        cae=cae,
        args=args,
        dataset=dataset,
        opt=opt,
    )

    checkpoint_callback = ModelCheckpoint(
        exp_folder,
        monitor="val_acc",
        mode="max",
        save_last=True,
        filename="uqgan",
    )

    time_estimator_callback = TimeEstimator(
        max_iterations=args["iterations"],
        logger=_log,
        interval="step",
        divider=1
        + args.get("discriminator_iterations", 1)
        + args.get("classifier_iterations", 1),
    )

    custom_checkpoint_io = CustomCheckpointIO()

    trainer = Trainer(
        default_root_dir=exp_folder,
        logger=sacred_logger,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, time_estimator_callback],
        plugins=[custom_checkpoint_io],
        max_steps=args["iterations"]
        * (
            1
            + args.get("discriminator_iterations", 1)
            + args.get("classifier_iterations", 1)
        ),
        max_epochs=-1,
        enable_progress_bar=args.get("enable_progress_bar", False),
        log_every_n_steps=5,
        check_val_every_n_epoch=None,
        val_check_interval=args.get("val_check_interval"),
    )

    ########################################
    #            Training
    ########################################

    trainer.fit(uqgan, train_dataloaders=trainloader, val_dataloaders=valloader)
