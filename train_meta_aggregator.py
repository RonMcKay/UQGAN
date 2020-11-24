from functools import partial
from os.path import exists, join

import numpy as np
from sacred import Experiment
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from cls_models import cls_models, load_cls_model, set_model_to_mode
from cls_models.base import BaseClassifier
from config import Config
from datasets import datasets, load_data
from logging_utils import log_config
from options import print_options
from utils import entropy, get_range, init_experiment, register_exp_folder

ex = Experiment("train_meta_aggregator", ingredients=(cls_models, datasets))
init_experiment(ex, mongo_observer=False)

device = None


@ex.capture
def gather_data(dataloader, classifier: BaseClassifier, args, _log, **kwargs):
    classifier.average_mc_samples = (
        False
        if getattr(classifier, "mc_dropout", 0.0) > 0 or classifier.method == "bayes"
        else True
    )
    class_targets = torch.empty((0,))
    outputs = None
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            class_targets = torch.cat((class_targets, y))

            out = classifier(x)

            if outputs is None:
                if isinstance(out, torch.Tensor):
                    out = [out]
                outputs = list(out)
            else:
                outputs = [torch.cat((i, j)) for i, j in zip(outputs, out)]

    class_probs, au, eu = outputs

    if classifier.method == "uqgan":

        if not classifier.average_mc_samples:
            wrong_predictions = class_probs.mean(0).argmax(1) != class_targets

            input_features = torch.cat(
                (
                    eu.mean(0).view(-1, 1),
                    au.mean(0).view(-1, 1),
                    class_probs.std(0).sum(1).view(-1, 1),
                ),
                dim=1,
            )
        else:
            wrong_predictions = class_probs.argmax(1) != class_targets

            input_features = torch.cat((eu.view(-1, 1), au.view(-1, 1)), dim=1)

        if args["include_preds"]:
            if not classifier.average_mc_samples:
                input_features = torch.cat((class_probs.mean(0), input_features), dim=1)
            else:
                input_features = torch.cat((class_probs, input_features), dim=1)
    elif classifier.method == "softmax":
        wrong_predictions = outputs[0].argmax(1) != class_targets
        if args["include_preds"]:
            input_features = class_probs
        else:
            input_features = class_probs.max(1)[0].view(-1, 1)
    elif classifier.method == "entropy":
        wrong_predictions = outputs[0].argmax(1) != class_targets
        input_features = au.view(-1, 1)
        if args["include_preds"]:
            input_features = torch.cat((class_probs, input_features), dim=1)
    elif classifier.method == "mc-dropout":
        wrong_predictions = outputs[0].argmax(1) != class_targets
        input_features = torch.cat(
            (
                entropy(class_probs.mean(0)).view(-1, 1),
                class_probs.std(0).sum(1).view(-1, 1),
            ),
            dim=1,
        )
        if args["include_preds"]:
            input_features = torch.cat(
                (
                    class_probs.mean(0),
                    input_features,
                ),
                dim=1,
            )
    elif classifier.method == "ensemble":
        return class_probs, class_targets
    elif classifier.method == "gen":
        wrong_predictions = outputs[0].argmax(1) != class_targets
        input_features = au.view(-1, 1)
        if args["include_preds"]:
            input_features = torch.cat((class_probs, input_features), dim=1)
    elif classifier.method == "bayes":
        wrong_predictions = outputs[0].argmax(1) != class_targets
        input_features = torch.cat((au.view(-1, 1), eu.view(-1, 1)), dim=1)
        if args["include_preds"]:
            input_features = torch.cat((class_probs, input_features), dim=1)

    targets = torch.zeros_like(wrong_predictions, dtype=torch.long)
    targets[wrong_predictions] = 1
    targets[class_targets == -1] = 2
    targets = targets.numpy()

    return input_features, targets


def target_transformation(target, ood=False):
    if ood:
        return -1
    else:
        return target


@ex.config
def config(dataset):
    args = dict(  # noqa: F841
        batch_size=256,
        gpu=0,
        save_folder=Config.root_save_folder,
        num_workers=8,
        ood_datasets=None,
        type="boosting",
        balance_datasets=False,
        exp_ids=None,
        exp_root=Config.root_save_folder,
        include_preds=True,
        ensemble_size=1,
    )


@ex.command(unobserved=True)
def options(args, dataset, cls_model):
    used_options = set(["datasets", "cls_models"])
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
    _ = register_exp_folder(args["save_folder"], _run)

    global device
    if torch.cuda.is_available() and args["gpu"] is not None:
        device = torch.device("cuda:{}".format(args["gpu"]))
    else:
        _log.info("Cuda is not available. Training will be done on CPU.")
        device = torch.device("cpu")

    traindat, _ = load_data(mode="train")
    indat, _ = load_data(mode="eval", target_transform=target_transformation)
    testdat, _ = load_data(
        mode="test", static=True, target_transform=target_transformation
    )

    ood_datasets = []
    ood_test_datasets = []

    if args["ood_datasets"] is None:
        raise ValueError("No OOD dataset specified")
    else:
        n_ood_datasets = len(args["ood_datasets"].split(","))
        for d in args["ood_datasets"].split(","):
            ood_cfg = dict(
                name=d,
                mode="eval",
                static=False,
                image_channels=dataset["cfg"]["image_channels"],
                target_transform=partial(target_transformation, ood=True),
            )
            ood_datasets.append(load_data(**ood_cfg)[0])

            ood_cfg["mode"] = "test"
            ood_cfg["static"] = True
            ood_test_datasets.append(load_data(**ood_cfg)[0])

            if args["balance_datasets"]:
                ood_datasets[-1] = Subset(
                    ood_datasets[-1],
                    torch.randperm(
                        len(ood_datasets[-1]),
                        generator=torch.Generator().manual_seed(42),
                    )[: len(indat) // n_ood_datasets],
                )

    # sample weights for an equal amount of samples from all provided OOD datasets
    ood_weights = []
    for ood_dataset in ood_datasets:
        ood_weights += [1.0 / (len(ood_dataset) * len(ood_datasets))] * len(ood_dataset)

    ooddat = ConcatDataset(ood_datasets)
    ood_test_dat = ConcatDataset(ood_test_datasets)

    traindat = ConcatDataset([indat, ooddat])

    trainloader = DataLoader(
        traindat,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    testloader = DataLoader(
        ConcatDataset([testdat, ood_test_dat]),
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
    )

    if isinstance(args["exp_ids"], str):
        exp_ids = get_range(args["exp_ids"])
    elif isinstance(args["exp_ids"], int):
        exp_ids = [args["exp_ids"]]
    else:
        exp_ids = args["exp_ids"]

    if args["exp_root"] is not None and exp_ids is not None:
        cls_checkpoints = []
        for id in exp_ids:
            if not exists(
                join(args["exp_root"], f"experiment_{id}", "classifier.ckpt")
            ):
                raise ValueError(
                    f"There is no checkpoint for experiment with id '{id}'."
                )
            cls_checkpoints.append(
                join(args["exp_root"], f"experiment_{id}", "classifier.ckpt")
            )
    elif isinstance(cls_model["cfg"]["checkpoint"], str):
        cls_checkpoints = [cls_model["cfg"]["checkpoint"]]
    elif isinstance(cls_model["cfg"]["checkpoint"], list):
        cls_checkpoints = cls_model["cfg"]["checkpoint"]
    else:
        raise ValueError(
            (
                "'checkpoint' has to be either of type 'str' or 'list' "
                f"but found '{type(cls_model['cfg']['checkpoint'])}'"
            )
        )

    ensemble_features = torch.empty((0,))
    ensemble_test_features = torch.empty((0,))
    confusion_matrices = []

    results = np.zeros(
        (
            len(cls_checkpoints)
            if args.get("ensemble_size") > 1
            else len(cls_checkpoints) // args.get("ensemble_size"),
            4,
        )
    )
    for cp_ind, cls_cp in enumerate(cls_checkpoints):
        _log.info(f"Evaluating checkpoint '{cls_cp}'")
        classifier = load_cls_model(checkpoint=cls_cp)
        classifier = classifier.to(device)
        set_model_to_mode(classifier, "eval")

        ########################################
        #            Predict metrics
        ########################################

        _log.info("Gathering data from trainloader")
        input_features, targets = gather_data(trainloader, classifier)

        _log.info("Gathering data from testloader")
        test_features, test_targets = gather_data(testloader, classifier)

        if args.get("ensemble_size") > 1:
            ensemble_features = torch.cat(
                (ensemble_features, input_features.unsqueeze(0))
            )
            ensemble_test_features = torch.cat(
                (ensemble_test_features, test_features.unsqueeze(0))
            )
            if (cp_ind + 1) % args.get("ensemble_size") == 0:
                if args["include_preds"]:
                    input_features = torch.cat(
                        (
                            ensemble_features.mean(0),
                            entropy(ensemble_features.mean(0)).view(-1, 1),
                            ensemble_features.std(0).sum(1).view(-1, 1),
                        ),
                        dim=1,
                    )
                else:
                    input_features = torch.cat(
                        (
                            entropy(ensemble_features.mean(0)).view(-1, 1),
                            ensemble_features.std(0).sum(1).view(-1, 1),
                        ),
                        dim=1,
                    )
                wrong_predictions = ensemble_features.mean(0).argmax(1) != targets

                if args["include_preds"]:
                    test_features = torch.cat(
                        (
                            ensemble_test_features.mean(0),
                            entropy(ensemble_test_features.mean(0)).view(-1, 1),
                            ensemble_test_features.std(0).sum(1).view(-1, 1),
                        ),
                        dim=1,
                    )
                else:
                    test_features = torch.cat(
                        (
                            entropy(ensemble_test_features.mean(0)).view(-1, 1),
                            ensemble_test_features.std(0).sum(1).view(-1, 1),
                        ),
                        dim=1,
                    )
                wrong_test_predictions = (
                    ensemble_test_features.mean(0).argmax(1) != test_targets
                )

                tmp_targets = torch.zeros_like(wrong_predictions, dtype=torch.long)
                tmp_targets[wrong_predictions] = 1
                tmp_targets[targets == -1] = 2
                tmp_targets = tmp_targets.numpy()
                targets = tmp_targets

                tmp_targets = torch.zeros_like(wrong_test_predictions, dtype=torch.long)
                tmp_targets[wrong_test_predictions] = 1
                tmp_targets[test_targets == -1] = 2
                tmp_targets = tmp_targets.numpy()
                test_targets = tmp_targets
            else:
                continue

        ########################################
        #            Training
        ########################################

        if args["type"] == "logistic":
            model = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
            )
            sample_weight = np.ones_like(targets, dtype=np.float64)
        elif args["type"] == "boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
            sample_weight = np.zeros_like(targets, dtype=np.float64)
            sample_weight[targets == 0] = 1.0 / np.sum(targets == 0)
            sample_weight[targets == 1] = 1.0 / np.sum(targets == 1)
            sample_weight[targets == 2] = 1.0 / np.sum(targets == 2)

        _log.info("Fitting model")
        model.fit(input_features, targets, sample_weight)

        ########################################
        #            Evaluate
        ########################################

        out = model.predict(test_features)
        correct_predictions = out == test_targets

        confusion_matrices.append(confusion_matrix(test_targets, out))

        results[
            cp_ind
            if args.get("ensemble_size") == 1
            else cp_ind // args.get("ensemble_size"),
            0,
        ] = np.mean(correct_predictions)
        results[
            cp_ind
            if args.get("ensemble_size") == 1
            else cp_ind // args.get("ensemble_size"),
            1,
        ] = np.mean(correct_predictions[test_targets == 0])
        results[
            cp_ind
            if args.get("ensemble_size") == 1
            else cp_ind // args.get("ensemble_size"),
            2,
        ] = np.mean(correct_predictions[test_targets == 1])
        results[
            cp_ind
            if args.get("ensemble_size") == 1
            else cp_ind // args.get("ensemble_size"),
            3,
        ] = np.mean(correct_predictions[test_targets == 2])

    result = np.mean(results, axis=0)
    result_std = np.std(results, axis=0)

    _log.info(f"Accuracy: {result[0]:.2%} +- {result_std[0]:.2%}")
    _log.info(f"Accuracy for TP: {result[1]:.2%} +- {result_std[1]:.2%}")
    _log.info(f"Accuracy for FP: {result[2]:.2%} +- {result_std[2]:.2%}")
    _log.info(f"Accuracy for OoD: {result[3]:.2%} +- {result_std[3]:.2%}")
    _log.info(
        f"Average Confusion Matrix:\n{np.mean(np.stack(confusion_matrices), axis=0)}"
    )
