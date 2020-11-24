import os
from os.path import abspath, exists, expanduser, join

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sacred import Experiment
from tabulate import tabulate
import torch
from torch.utils.data import ConcatDataset, DataLoader

from cls_models import cls_models, load_cls_model
from config import Config
from datasets import datasets, load_data
from eval.binary import aupr, auroc, ece, fprxtpr
from logging_utils import log_config
from options import print_options
from utils import (
    entropy,
    extract_exp_id_from_path,
    format_int_list,
    get_accelerator_device,
    get_range,
    init_experiment,
)

ex = Experiment(
    "evaluate OOD detection deep ensembles", ingredients=[datasets, cls_models]
)
init_experiment(ex, mongo_observer=False)


@ex.config
def config():
    args = dict(  # noqa: F841
        gpu=0,
        batch_size=64,
        num_workers=8,
        ood_datasets=None,
        export=False,
        root_metric_export_folder=Config.root_metric_export_folder,
        save_folder=Config.root_save_folder,
        exp_ids=None,
        ensemble_size=5,
        method_overwrite="entropy",
    )


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


@ex.automain
def main(args, cls_model, dataset, _run, _log):
    log_config(_run, _log)

    ########################################
    #              Set devices
    ########################################

    accelerator, devices = get_accelerator_device(args["gpu"])

    ########################################
    #       Load dataset and model
    ########################################

    indistdat, _ = load_data()
    _log.info(f"{len(indistdat)} In-Samples of '{dataset['cfg']['name']}'")

    indistloader = DataLoader(
        indistdat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

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
            if "image_size" in ood_cfg:
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

    if len(checkpoints) % args["ensemble_size"] != 0:
        raise ValueError(
            (
                f"Number of checkpoints ({len(checkpoints)}) is not divisible by "
                f"the ensemble size ({args['ensemble_size']})"
            )
        )

    results = np.zeros(
        (len(checkpoints) // args["ensemble_size"], len(ood_datasets) + 2, 4)
    )
    results_acc = np.zeros(len(checkpoints) // args["ensemble_size"])
    results_ece = np.zeros(len(checkpoints) // args["ensemble_size"])
    columns = ["auroc", "aupr-in", "aupr-out", "fpr95tpr"]
    index = args["ood_datasets"].split(",") + ["all", "succ/fail"]

    trainer = Trainer(
        logger=False,
        enable_progress_bar=args.get("enable_progress_bar", True),
        max_epochs=-1,
        accelerator=accelerator,
        devices=devices,
    )

    for ensemble_index, sub_checkpoints in enumerate(
        [
            checkpoints[i : i + args["ensemble_size"]]
            for i in range(0, len(checkpoints), args["ensemble_size"])
        ]
    ):
        list_in_class_probs = []
        list_out_class_probs = []
        for sub_cp_index, sub_cp in enumerate(sub_checkpoints):
            _log.info(
                (
                    f"Predicting with ensemble '{ensemble_index}' "
                    f"- Sub Checkpoint '{sub_cp_index}'"
                )
            )
            classifier = load_cls_model(
                checkpoint=sub_cp, cp_overwrite={"method": args["method_overwrite"]}
            )

            ########################################
            #              Predict
            ########################################

            _log.info("Predicting in-distribution set...")
            if sub_cp_index == (len(sub_checkpoints) - 1):
                class_target = torch.empty((0,))
                for _, y in indistloader:
                    class_target = torch.cat((class_target, y))

            tmp_out = trainer.predict(classifier, indistloader)
            class_probs, *_ = [torch.cat(o) for o in zip(*tmp_out)]

            list_in_class_probs.append(class_probs)

            _log.info("Predicting ood sets...")
            out_class_probs = torch.empty((0,))
            ooddat = ConcatDataset(ood_datasets)
            oodloader = DataLoader(
                ooddat,
                batch_size=args["batch_size"],
                shuffle=False,
                num_workers=args["num_workers"],
            )

            tmp_out = trainer.predict(classifier, oodloader)
            out_class_probs = torch.cat([o[0] for o in tmp_out])

            list_out_class_probs.append(out_class_probs)

        _log.info(f"Evaluating ensemble '{ensemble_index}'")

        in_class_probs = torch.stack(list_in_class_probs).mean(0)
        class_pred = in_class_probs.argmax(1)
        wrong_predictions = (class_pred != class_target).float()

        if classifier.method == "entropy":
            in_dist_metric = entropy(in_class_probs)
        elif classifier.method == "softmax":
            in_dist_metric = in_class_probs.max(1)[0]

        ####################################

        out_class_probs = torch.stack(list_out_class_probs).mean(0)

        if classifier.method == "entropy":
            ood_metric = entropy(out_class_probs)
        elif classifier.method == "softmax":
            ood_metric = out_class_probs.max(1)[0]

        ####################################

        cumulative_dataset_sizes = np.cumsum([len(d) for d in ood_datasets])
        ood_metrics = [ood_metric[: len(ood_datasets[0])]]
        for i in range(1, len(cumulative_dataset_sizes)):
            ood_metrics.append(
                ood_metric[
                    cumulative_dataset_sizes[i - 1] : cumulative_dataset_sizes[i]
                ]
            )

        ####################################

        _log.info("Evaluating...")
        for i, d in enumerate(args["ood_datasets"].split(",")):
            pred = torch.cat((in_dist_metric, ood_metrics[i]))
            pred = (pred - pred.min()) / (max(pred.max() - pred.min(), 1e-16))

            if classifier.method == "entropy":
                target = torch.cat(
                    (torch.zeros_like(in_dist_metric), torch.ones_like(ood_metrics[i]))
                )
                results[ensemble_index, i, 0] = auroc(pred=pred, target=target)
                results[ensemble_index, i, 1] = aupr(pred=1 - pred, target=1 - target)
                results[ensemble_index, i, 2] = aupr(pred=pred, target=target)
                results[ensemble_index, i, 3] = fprxtpr(
                    pred=1 - pred, target=1 - target
                )
            elif classifier.method == "softmax":
                target = torch.cat(
                    (torch.ones_like(in_dist_metric), torch.zeros_like(ood_metrics[i]))
                )
                results[ensemble_index, i, 0] = auroc(pred=pred, target=target)
                results[ensemble_index, i, 1] = aupr(pred=pred, target=target)
                results[ensemble_index, i, 2] = aupr(pred=1 - pred, target=1 - target)
                results[ensemble_index, i, 3] = fprxtpr(pred=pred, target=target)

        metrics_all = torch.cat([in_dist_metric] + ood_metrics)
        metrics_all = (metrics_all - metrics_all.min()) / (
            max(metrics_all.max() - metrics_all.min(), 1e-16)
        )

        if classifier.method == "entropy":
            targets_all = torch.cat(
                (
                    torch.zeros_like(in_dist_metric),
                    torch.ones_like(torch.cat(ood_metrics)),
                )
            )
            results[ensemble_index, len(ood_datasets), 0] = auroc(
                metrics_all, targets_all
            )
            results[ensemble_index, len(ood_datasets), 1] = aupr(
                1 - metrics_all, 1 - targets_all
            )
            results[ensemble_index, len(ood_datasets), 2] = aupr(
                metrics_all, targets_all
            )
            results[ensemble_index, len(ood_datasets), 3] = fprxtpr(
                1 - metrics_all, 1 - targets_all
            )

            results[ensemble_index, len(ood_datasets) + 1, 0] = auroc(
                in_dist_metric, wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 1] = aupr(
                1 - in_dist_metric, 1 - wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 2] = aupr(
                in_dist_metric, wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 3] = fprxtpr(
                1 - in_dist_metric, 1 - wrong_predictions, autonorm=True
            )

        elif classifier.method == "softmax":
            targets_all = torch.cat(
                (
                    torch.ones_like(in_dist_metric),
                    torch.zeros_like(torch.cat(ood_metrics)),
                )
            )
            results[ensemble_index, len(ood_datasets), 0] = auroc(
                metrics_all, targets_all
            )
            results[ensemble_index, len(ood_datasets), 1] = aupr(
                metrics_all, targets_all
            )
            results[ensemble_index, len(ood_datasets), 2] = aupr(
                1 - metrics_all, 1 - targets_all
            )
            results[ensemble_index, len(ood_datasets), 3] = fprxtpr(
                metrics_all, targets_all
            )

            results[ensemble_index, len(ood_datasets) + 1, 0] = auroc(
                in_dist_metric, 1 - wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 1] = aupr(
                in_dist_metric, 1 - wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 2] = aupr(
                1 - in_dist_metric, wrong_predictions, autonorm=True
            )
            results[ensemble_index, len(ood_datasets) + 1, 3] = fprxtpr(
                in_dist_metric, 1 - wrong_predictions, autonorm=True
            )

        results_ece[ensemble_index] = ece(
            class_pred, class_target, class_probs.max(1)[0]
        )

        results_acc[ensemble_index] = (class_pred == class_target).float().mean().item()

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
                f"results_{args['type']}_deep_ensembles_oodd_succfail_{exp_id}.csv",
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
                f"results_{args['type']}_deep_ensembles_accuracy_ece_{exp_id}.csv",
            ),
            index_label="datasets",
        )

    return result
