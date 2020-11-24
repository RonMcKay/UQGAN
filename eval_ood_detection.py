from logging import Logger
import os
from os.path import abspath, exists, expanduser, join

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sacred import Experiment
from tabulate import tabulate
import torch
from torch.utils.data import ConcatDataset, DataLoader

from cls_models import cls_models, load_cls_model, set_model_to_mode
from cls_models.base import BaseClassifier
from config import Config
from datasets import datasets, load_data
from eval.binary import aupr, auroc, ece, fprxtpr
from logging_utils import log_config
from options import print_options
from utils import (
    extract_exp_id_from_path,
    format_int_list,
    get_accelerator_device,
    get_range,
    init_experiment,
)

ex = Experiment("evaluate OOD detection", ingredients=[datasets, cls_models])
init_experiment(ex, mongo_observer=False)


@ex.config
def config(cls_model):
    args = dict(  # noqa: F841
        gpu=0,
        batch_size=64,
        num_workers=8,
        ood_datasets=None,
        export=False,
        root_metric_export_folder=Config.root_metric_export_folder,
        save_folder=Config.root_save_folder,
        exp_ids=None,
        method_overwrite=None,
    )

    if cls_model["cfg"].get("mc_dropout", 0.0) > 0:
        args["mc_samples"] = 50


@ex.command(unobserved=True)
def options(args, dataset, cls_model):
    used_options = set(["enable_progress_bar", "cls_models", "datasets"])
    used_options = used_options.union(
        set(
            list(args.keys())
            + list(dataset["cfg"].keys())
            + list(cls_model["cfg"].keys())
        )
    )

    print_options(used_options)


def eval_classifier(classifier, indist_dataset, ood_datasets, args, log, trainer=None):
    set_model_to_mode(classifier, "eval")
    indist_loader = DataLoader(
        indist_dataset,
        batch_size=args.get("batch_size", 64),
        shuffle=False,
        num_workers=args.get("num_workers", 0),
    )

    results = np.zeros((len(ood_datasets) + 2, 4))

    ########################################
    #              Evaluate
    ########################################

    log.debug("Predicting in-distribution set...")
    if trainer is None:
        class_target = torch.empty((0,))
        class_probs = torch.empty((0,))
        in_au = torch.empty((0,))
        in_eu = torch.empty((0,))
        with torch.no_grad():
            for x, y in indist_loader:
                x = x.to(classifier.device)
                class_target = torch.cat((class_target, y))

                tmp_out = classifier(x)
                class_probs = torch.cat((class_probs, tmp_out[0].cpu()))
                in_au = torch.cat((in_au, tmp_out[1].cpu()))
                in_eu = torch.cat((in_eu, tmp_out[2].cpu()))
    else:
        tmp_out = trainer.predict(classifier, indist_loader)
        class_probs, in_au, in_eu = [torch.cat(o).cpu() for o in zip(*tmp_out)]

        class_target = torch.empty((0,))
        for _, y in indist_loader:
            class_target = torch.cat((class_target, y))

    class_pred = class_probs.argmax(1)

    wrong_predictions = (class_pred != class_target).float()

    ####################################

    log.debug("Predicting ood sets...")

    oodloader = DataLoader(
        ConcatDataset(ood_datasets),
        shuffle=False,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
    )

    if trainer is None:
        out_eu = torch.empty((0,))
        with torch.no_grad():
            for x, y in oodloader:
                x = x.to(classifier.device)

                *_, tmp_eu = classifier(x)
                out_eu = torch.cat((out_eu, tmp_eu.cpu()))
    else:
        tmp_out = trainer.predict(classifier, oodloader)
        *_, out_eu = [torch.cat(o).cpu() for o in zip(*tmp_out)]

    cumulative_dataset_sizes = np.cumsum([len(d) for d in ood_datasets])
    out_eus = [out_eu[: len(ood_datasets[0])]]
    for i in range(1, len(cumulative_dataset_sizes)):
        out_eus.append(
            out_eu[cumulative_dataset_sizes[i - 1] : cumulative_dataset_sizes[i]]
        )

    ####################################

    log.debug("Evaluating...")
    for i, d in enumerate(args["ood_datasets"].split(",")):
        pred = torch.cat((in_eu, out_eus[i]))
        pred = (pred - pred.min()) / (max(pred.max() - pred.min(), 1e-16))

        target = torch.cat((torch.zeros_like(in_eu), torch.ones_like(out_eus[i])))

        results[i, 0] = auroc(pred=pred, target=target)
        results[i, 1] = aupr(pred=1 - pred, target=1 - target)  # AUPR-In
        results[i, 2] = aupr(pred=pred, target=target)  # AUPR-Out
        results[i, 3] = fprxtpr(pred=1 - pred, target=1 - target)

    metrics_all = torch.cat([in_eu] + out_eus)
    metrics_all = (metrics_all - metrics_all.min()) / (
        max(metrics_all.max() - metrics_all.min(), 1e-16)
    )

    targets_all = torch.cat(
        (
            torch.zeros_like(in_eu),
            torch.ones_like(torch.cat(out_eus)),
        )
    )

    results[len(ood_datasets), 0] = auroc(metrics_all, targets_all)
    results[len(ood_datasets), 1] = aupr(1 - metrics_all, 1 - targets_all)  # AUPR-In
    results[len(ood_datasets), 2] = aupr(metrics_all, targets_all)  # AUPR-Out
    results[len(ood_datasets), 3] = fprxtpr(1 - metrics_all, 1 - targets_all)

    in_au = (in_au - in_au.min()) / max(in_au.max() - in_au.min(), 1e-16)

    results[len(ood_datasets) + 1, 0] = auroc(in_au, wrong_predictions, autonorm=True)
    results[len(ood_datasets) + 1, 1] = aupr(
        1 - in_au, 1 - wrong_predictions, autonorm=True
    )  # AUPR-S
    results[len(ood_datasets) + 1, 2] = aupr(
        in_au, wrong_predictions, autonorm=True
    )  # AUPR-F
    results[len(ood_datasets) + 1, 3] = fprxtpr(in_au, wrong_predictions, autonorm=True)

    result_ece = ece(class_pred, class_target, class_probs.max(1)[0])
    result_acc = (class_pred == class_target).float().mean().item()

    return results, result_ece, result_acc


@ex.automain
def main(args, cls_model, dataset, _run, _log: Logger):
    log_config(_run, _log)

    ########################################
    #              Set devices
    ########################################

    accelerator, devices = get_accelerator_device(args["gpu"])

    ########################################
    #       Load dataset and model
    ########################################
    indistdat, _ = load_data()
    _log.debug(f"{len(indistdat)} In-Samples of '{dataset['cfg']['name']}'")

    ood_datasets = []

    if args["ood_datasets"] is None:
        raise ValueError("No OOD dataset specified")
    else:
        for d in args["ood_datasets"].split(","):
            ood_cfg = dict(
                name=d,
                mode=dataset["cfg"]["mode"],
                static=True,
                image_channels=dataset["cfg"]["image_channels"],
            )
            if "image_size" in dataset["cfg"]:
                ood_cfg["image_size"] = dataset["cfg"]["image_size"]
            ood_datasets.append(load_data(**ood_cfg)[0])
            _log.info(f"{len(ood_datasets[-1])} OOD-Samples of '{d}'")

    ########################################
    #           Load checkpoints
    ########################################

    if isinstance(args["exp_ids"], str):
        exp_ids = get_range(args["exp_ids"])
    elif isinstance(args["exp_ids"], int):
        exp_ids = [args["exp_ids"]]
    else:
        exp_ids = args["exp_ids"]

    if args["save_folder"] is not None and exp_ids is not None:
        checkpoints = []
        for id in exp_ids:
            if not exists(
                join(args["save_folder"], f"experiment_{id}", "classifier.ckpt")
            ):
                raise ValueError(
                    f"There is no checkpoint for experiment with id '{id}''."
                )
            checkpoints.append(
                join(args["save_folder"], f"experiment_{id}", "classifier.ckpt")
            )
    elif isinstance(cls_model["cfg"]["checkpoint"], str):
        checkpoints = [cls_model["cfg"]["checkpoint"]]
    elif isinstance(cls_model["cfg"]["checkpoint"], list):
        checkpoints = cls_model["cfg"]["checkpoint"]
    else:
        raise ValueError(
            (
                "'checkpoint' has to be either of type 'str' or 'list' "
                f"but found '{type(cls_model['cfg']['checkpoint'])}'"
            )
        )

    results = np.zeros((len(checkpoints), len(ood_datasets) + 2, 4))
    results_acc = np.zeros(len(checkpoints))
    results_ece = np.zeros(len(checkpoints))
    columns = ["auroc", "aupr-in", "aupr-out", "fpr95tpr"]
    index = args["ood_datasets"].split(",") + ["all", "succ/fail"]

    trainer = Trainer(
        logger=False,
        enable_progress_bar=args.get("enable_progress_bar", True),
        max_epochs=-1,
        accelerator=accelerator,
        devices=devices,
    )

    for cp_ind, cp in enumerate(checkpoints):
        _log.info(f"Evaluating checkpoint '{cp}'")
        classifier = load_cls_model(
            checkpoint=cp,
            cp_overwrite={"method": args["method_overwrite"]}
            if args["method_overwrite"] is not None
            else None,
        )  # type: BaseClassifier

        tmp_results, tmp_ece, tmp_acc = eval_classifier(
            classifier=classifier,
            indist_dataset=indistdat,
            ood_datasets=ood_datasets,
            args=args,
            log=_log,
            trainer=trainer,
        )

        results[cp_ind] = tmp_results
        results_ece[cp_ind] = tmp_ece
        results_acc[cp_ind] = tmp_acc

    result = np.mean(results, axis=0)
    result_std = np.std(results, axis=0)
    acc_mean = np.mean(results_acc)
    acc_std = np.std(results_acc)
    ece_mean = np.mean(results_ece)
    ece_std = np.std(results_ece)

    _log.info(f"In-Distribution class accuracy: {acc_mean:.2%} +- {acc_std:.2%}")
    _log.info(f"Expected Calibration Error: {ece_mean:.2%} +- {ece_std:.2%}")

    _log.info(
        "\n" * 2
        + tabulate(
            {
                "OOD Dataset": args["ood_datasets"].split(",") + ["all", "succ/fail"],
                "AUROC": [
                    f"{result[i, 0]:.2%} +- {result_std[i, 0]:.2%}"
                    for i in range(result.shape[0])
                ],
                "AUPR-In": [
                    f"{result[i, 1]:.2%} +- {result_std[i, 1]:.2%}"
                    for i in range(result.shape[0])
                ],
                "AUPR-Out": [
                    f"{result[i, 2]:.2%} +- {result_std[i, 2]:.2%}"
                    for i in range(result.shape[0])
                ],
                "FPR @ 95% TPR": [
                    f"{result[i, 3]:.2%} +- {result_std[i, 3]:.2%}"
                    for i in range(result.shape[0])
                ],
            },
            headers="keys",
            floatfmt="3.2%",
            tablefmt="github",
        )
        + "\n"
    )

    if args["export"]:
        if not exists(abspath(expanduser(args["root_metric_export_folder"]))):
            os.makedirs(abspath(expanduser(args["root_metric_export_folder"])))

        if exp_ids is not None:
            exp_id = format_int_list(exp_ids)
        else:
            exp_id = extract_exp_id_from_path(cls_model["cfg"]["checkpoint"])

        pd.DataFrame(result, index=index, columns=columns).to_csv(
            join(
                args["root_metric_export_folder"],
                f"results_{args['type']}_baseline_oodd_succfail_{exp_id}.csv",
            ),
            index_label="datasets",
        )

        pd.DataFrame(
            np.concatenate((acc_mean.reshape(-1, 1), ece_mean.reshape(-1, 1)), axis=1),
            index=["all"],
            columns=["accuracy", "ece"],
        ).to_csv(
            join(
                args["root_metric_export_folder"],
                f"results_{args['type']}_baseline_accuracy_ece_{exp_id}.csv",
            ),
            index_label="datasets",
        )

    return result
