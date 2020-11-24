from typing import List, Set, Tuple, Union

from tabulate import tabulate

from cae_models import CAE_MODELS
from cls_models import CLS_MODELS
from cls_models.base import METHODS
from config import Config
from datasets import default_configs
from gan_models import GAN_MODELS
from utils import AVAILABLE_REG_TYPES

options: List[Tuple[str, str, str]]

options = [
    ("batch_size", "int", "Number of samples in a batch"),
    ("balance_datasets", "bool", "Whether to balance the datasets during training."),
    ("beta", "float", "Weight used for the Kullback-Leibler Divergence term"),
    (
        "cae_models",
        "str",
        f"Available conditional Auto-Encoders: {', '.join(CAE_MODELS)}",
    ),
    (
        "check_val_every_n_epoch",
        "int",
        "Number of epochs to run the evaluation routine after",
    ),
    ("checkpoint", "str", "Path to a model checkpoint"),
    (
        "cl_dim",
        "int",
        "Number of classes in the dataset. This is for internal purposes. "
        "Do not change this.",
    ),
    (
        "classifier_iterations",
        "int",
        "Number of classifier iterations per generator iteration",
    ),
    ("cls_models", "str", f"Available classifiers: {', '.join(CLS_MODELS)}"),
    (
        "enable_progress_bar",
        "bool",
        "Whether to display a progress bar during training/inference",
    ),
    (
        "datasets",
        "str",
        f"Available datasets: {', '.join(c['name'] for c in default_configs.values())}",
    ),
    (
        "discriminator_iterations",
        "int",
        "Number of discriminator iterations per generator iteration",
    ),
    ("epochs", "int", "Number of epochs to train"),
    (
        "ensemble_size",
        "int",
        "Size of the ensemble which should be used for evaluation",
    ),
    (
        "exp_root",
        "str",
        "Directory where to load the checkpoints from. "
        "Might be different from 'save_folder' but defaults to the same.",
    ),
    ("export", "bool", "Whether to export the metric results to a file."),
    ("gan_models", "str", f"Available GAN architectures: {', '.join(GAN_MODELS)}"),
    ("gpu", "int or list[int]", "GPU id or list of GPU ids to put the models on"),
    (
        "image_channels",
        "int",
        "Number of image channels in the dataset. "
        "You will probably not need this so do not change it.",
    ),
    (
        "image_size",
        "int",
        "Size of the square image of the dataset. "
        "You will probably not need this so do not change it.",
    ),
    (
        "include_preds",
        "bool",
        "Whether to include the predicted class "
        "probabilities into the meta aggregator",
    ),
    (
        "input_size",
        "int",
        "Similar to 'image_size' but for toy datasets which are 2-dimensional.",
    ),
    ("iterations", "int", "Number of generator iterations"),
    ("latent_dim", "int", "Latent dimension of the Auto-Encoder or Generator."),
    ("lambda_cl_loss", "float", "Weight for the classification loss"),
    ("lambda_gp", "float", "Weight for the wasserstein gradient penalty"),
    ("lambda_real_ood", "float", "Weight for the real part of the ood loss"),
    ("lambda_reg_loss", "float", "Weight for the low dimensional regularization loss"),
    ("lr", "float", "Standard learning rate for all optimizers"),
    (
        "lr_cls",
        "float",
        "Learning rate of the classifier's optimizer. Overwrites 'lr'.",
    ),
    (
        "lr_disc",
        "float",
        "Learning rate of the discriminator's optimizer. Overwrites 'lr'.",
    ),
    (
        "lr_disc_image",
        "float",
        "Learning rate of the image space discriminator's optimizer. Overwrites 'lr'.",
    ),
    (
        "lr_disc_latent",
        "float",
        "Learning rate of the latent space discriminator's optimizer. Overwrites 'lr'.",
    ),
    (
        "lr_gen",
        "float",
        "Learning rate of the generator's optimizer. Overwrites 'lr'.",
    ),
    (
        "lr_vae",
        "float",
        "Learning rate of the variational Auto-Encoder's optimizer. Overwrites 'lr'.",
    ),
    (
        "mc_dropout",
        "int",
        "Probability for a unit to be dropped. 0 disables mc_dropout.",
    ),
    (
        "mc_samples",
        "int",
        "Number of samples used for one forward pass during training. "
        "Evaluation always uses 50 samples.",
    ),
    (
        "method",
        "str",
        (
            "Method this classifier uses for prediction/uncertainty quantification. "
            "Valid options: "
        )
        + ", ".join(METHODS),
    ),
    (
        "method_overwrite",
        "str",
        "Useful for overwriting a method stored in a checkpoint, e.g. if you trained "
        "a classic softmax classifier but want to evaluate with entropy as uncertainty",
    ),
    ("min_lr", "float", "Minimum learning rate for all learning rate schedulers"),
    (
        "min_lr_cls",
        "float",
        "Minimum learning rate used for the scheduler of the "
        "classifier learning rate. Overwrites 'min_lr'.",
    ),
    (
        "min_lr_disc",
        "float",
        "Minimum learning rate used for the scheduler of the "
        "discriminator learning rate. Overwrites 'min_lr'.",
    ),
    (
        "min_lr_gen",
        "float",
        "Minimum learning rate used for the scheduler of the "
        "generator learning rate. Overwrites 'min_lr'.",
    ),
    ("mode", "str", "Mode of the datasets. Available modes: train, test, eval"),
    ("n_samples", "int", "Number of samples in the toy dataset."),
    ("name", "str", "Name of a dataset or model."),
    ("num_sample_images", "int", "Number of images generated for debugging purposes"),
    ("num_workers", "int", "Number of threads used for data loading"),
    (
        "ood_datasets",
        "str",
        "List of Out-of-Distribution datasets to use for evaluation",
    ),
    (
        "reg_type",
        "str",
        "Type of low dimensional regularizer. "
        f"Available types: {', '.join(AVAILABLE_REG_TYPES)}",
    ),
    (
        "root_metric_export_folder",
        "str",
        "If 'export=True' the metric results will be exported to this folder",
    ),
    (
        "sample_every_n",
        "int",
        "Number of steps after which sample images should be generated",
    ),
    (
        "save_folder",
        "str",
        "Root path to create the individual experiment run folders in",
    ),
    (
        "static",
        "bool",
        "Whether to apply image augmentations or not. "
        "Should be turned of for evaluation.",
    ),
    (
        "type",
        "str",
        "Type of meta aggregator. Available types: logistic, boosting",
    ),
    (
        "vae_iterations",
        "int",
        "Number of training steps of the variational Auto-Encoder "
        "for each generator step.",
    ),
    (
        "val_check_interval",
        "int",
        "Number of generator iterations after which to run the evaluation routine",
    ),
    ("weight_decay", "float", "Standard weight decay used for all optimizers"),
    (
        "weight_decay_cls",
        "float",
        "Weight decay used by the optimizer of the classifier. "
        "Overwrites 'weight_decay'.",
    ),
    (
        "weight_decay_disc",
        "float",
        "Weight decay used by the optimizer of the discriminator. "
        "Overwrites 'weight_decay'.",
    ),
    (
        "weight_decay_disc_image",
        "float",
        "Weight decay used by the optimizer of the image space discriminator. "
        "Overwrites 'weight_decay'.",
    ),
    (
        "weight_decay_disc_latent",
        "float",
        "Weight decay used by the optimizer of the latent space discriminator. "
        "Overwrites 'weight_decay'.",
    ),
    (
        "weight_decay_gen",
        "float",
        "Weight decay used by the optimizer of the generator. "
        "Overwrites 'weight_decay'.",
    ),
]


def print_options(subset: Union[List[str], Set[str]] = None):
    printed_options = options
    if subset is not None:
        printed_options = [o for o in printed_options if o[0] in subset]
    printed_options.sort(key=lambda x: x[0])
    if printed_options:
        print(
            tabulate(
                printed_options,
                headers=["Option", "Type", "Description"],
                tablefmt="grid",
                maxcolwidths=Config.max_col_width,
            )
        )


if __name__ == "__main__":

    print(
        tabulate(
            options,
            headers=["Option", "Type", "Description"],
            tablefmt="grid",
            maxcolwidths=Config.max_col_width,
        )
    )
