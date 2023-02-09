# flake8: noqa
import os
from os.path import abspath, exists, expanduser, join
from typing import Dict, List, Union

import matplotlib

from config import Config

font = {
    "family": "Times New Roman",
    "size": 10,
}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

data_root = Config.root_metric_export_folder
plot_root = Config.root_plot_export_folder
if not exists(plot_root):
    os.makedirs(plot_root)


def plot_parameter_study(
    experiment_ids: Dict[str, List[int]],
    x_values: List[Union[float, int]],
    filename: str,
    x_label: str,
    x_scale: str = "linear",
    x_scale_kwargs: Dict = {},
    **kwargs,
):
    ncols = 2
    nrows = len(experiment_ids.keys())
    ood_legend_labels_1 = ["Accuracy", "AUROC", "AUPR-Out", "AUROC-S/F"]
    ood_legend_labels_2 = ["AUPR-In", "FPR@95TPR", "ECE"]
    colors = list(np.arange(len(ood_legend_labels_1) + len(ood_legend_labels_2)))
    my_cmap = plt.get_cmap("tab10")

    fig: matplotlib.figure.Figure
    fig, ax_all = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6), sharex="col")

    for d_ind, ((dataset_name, exp_ids), ax) in enumerate(
        zip(experiment_ids.items(), np.split(ax_all, nrows, axis=0))
    ):
        ax = ax.flatten()
        oodd_succfail_dataframes: Dict[str, pd.DataFrame] = {}
        for id in exp_ids:
            oodd_succfail_dataframes[id] = pd.read_csv(
                join(data_root, f"results_classifier_oodd_succfail_{id}.csv"),
                sep=",",
                index_col=0,
            )

        acc_dataframes: Dict[str, pd.DataFrame] = {}
        for id in exp_ids:
            acc_dataframes[id] = pd.read_csv(
                join(data_root, f"results_classifier_accuracy_ece_{id}.csv"),
                sep=",",
                index_col=0,
            )

        oodd_array = np.array(
            [df.loc["all"] for df in oodd_succfail_dataframes.values()]
        )
        succfail_array = np.array(
            [df.loc["succ/fail"] for df in oodd_succfail_dataframes.values()]
        )
        acc_array = np.array([df.loc["all"] for df in acc_dataframes.values()])

        first_array = (
            np.array(
                [
                    acc_array[:, 0],
                    oodd_array[:, 0],
                    oodd_array[:, 2],
                    succfail_array[:, 0],
                ]
            )
            * 100
        )
        first_min = np.min(first_array)
        i = 1
        while np.abs(first_array.mean() - first_min) > 25:
            first_min = first_array.flatten()[np.argsort(first_array, axis=None)[i]]
            i += 1
        first_max = np.max(first_array)

        first_diff = first_max - first_min
        first_min = first_min - 0.05 * first_diff
        first_max = first_max + 0.05 * first_diff

        second_array = (
            np.array([oodd_array[:, 1], oodd_array[:, 3], acc_array[:, 1]]) * 100
        )
        second_min = np.min(second_array)
        second_max = np.max(second_array)
        second_diff = second_max - second_min

        second_min = second_min - 0.05 * second_diff
        second_max = second_max + 0.05 * second_diff

        legend_handles = []
        for i, y in enumerate(first_array):
            legend_handles.append(
                ax[0].plot(
                    x_values,
                    y,
                    "-o",
                    color=my_cmap(colors[i]),
                    label=ood_legend_labels_1[i],
                )[0]
            )

        for i, y in enumerate(second_array, start=4):
            legend_handles.append(
                ax[1].plot(
                    x_values,
                    y,
                    "-o",
                    color=my_cmap(colors[i]),
                    label=ood_legend_labels_2[i - 4],
                )[0]
            )

        ax[0].grid(True)
        ax[0].set_xscale(x_scale, **x_scale_kwargs)
        ax[0].set_ylabel(r"Metric in \%")
        ax[0].set_ylim(first_min, first_max)

        ax[1].grid(True)
        ax[1].set_xscale(x_scale, **x_scale_kwargs)
        ax[1].set_ylabel(
            f"{{\\large \\textbf{{{dataset_name}}}}}\n\nMetric in \\%", rotation=-90
        )
        ax[1].yaxis.set_label_coords(1.2, 0.5)

        if d_ind == 0:
            ax[1].legend(
                legend_handles,
                [lh.get_label() for lh in legend_handles],
                fontsize=8.5,
                ncol=4,
                fancybox=True,
                handletextpad=0.2,
                columnspacing=0.3,
            )
        ax[1].set_ylim(second_min, second_max)
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()

        if d_ind == len(experiment_ids.keys()) - 1:
            ax[0].set_xlabel(x_label)
            ax[0].set_xticks(kwargs.get("xticks", x_values))
            ax[0].set_xticklabels(
                [str(i) for i in kwargs.get("xtick_labels", x_values)]
            )

            ax[1].set_xticks(kwargs.get("xticks", x_values))
            ax[1].set_xticklabels(
                [str(i) for i in kwargs.get("xtick_labels", x_values)]
            )
            ax[1].set_xlabel(x_label)

    fig.suptitle(kwargs.get("title", ""), y=0.93)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    plt.savefig(filename, bbox_inches="tight")


########################################################################################
print("Plotting 'lambda_cl_loss' parameter study")

x_values = list(range(9))
xtick_labels = ["0"] + [str(2**i) for i in range(8)]

plot_parameter_study(
    experiment_ids={
        "MNIST": [488, 509] + list(range(489, 496)),
        "CIFAR10": list(range(952, 961)),
    },
    x_values=x_values,
    filename=join(plot_root, "lambda_cl_loss_parameter_study.pdf"),
    x_label=r"$\lambda_{\textrm{cl}}$",
    xtick_labels=xtick_labels,
    title=r"\textbf{Parameter study for $\bm{\lambda_{\textrm{cl}}}$}",
)

########################################################################################
print("Plotting 'latent_dim' parameter study")

x_values = range(8)
xtick_labels = [str(2**i) for i in range(2, 10)]

plot_parameter_study(
    experiment_ids={
        "MNIST": list(range(575, 583)),
        "CIFAR10": list(range(995, 1003)),
    },
    x_values=x_values,
    filename=join(plot_root, "latent_dim_parameter_study.pdf"),
    x_label="Latent Dimension",
    xtick_labels=xtick_labels,
    title=r"\textbf{Parameter study for the latent dimension}",
)

########################################################################################
print("Plotting 'lambda_reg_loss' parameter study")

x_values = range(9)
xtick_labels = ["0"] + [str(2**i) for i in range(8)]

plot_parameter_study(
    experiment_ids={
        "MNIST": list(range(519, 522)) + [1003] + list(range(523, 528)),
        "CIFAR10": list(range(1014, 1023)),
    },
    x_values=x_values,
    filename=join(plot_root, "lambda_reg_loss_parameter_study.pdf"),
    x_label=r"$\lambda_{\textrm{reg}}$",
    xtick_labels=xtick_labels,
    title=r"\textbf{Parameter study for $\bm{\lambda_{\textrm{reg}}}$}",
)

########################################################################################
print("Plotting 'lambda_real' parameter study")

x_values = [float(i) / 10 for i in range(11)]

plot_parameter_study(
    experiment_ids={
        "MNIST": list(range(540, 551)),
        "CIFAR10": list(range(971, 982)),
    },
    x_values=x_values,
    filename=join(plot_root, "lambda_real_parameter_study.pdf"),
    x_label=r"$\lambda_{\textrm{real}}$",
    title=r"\textbf{Parameter study for $\bm{\lambda_{\textrm{real}}}$}",
)

########################################################################################
# print("Plotting 'reg_type' parameter study")

# experiment_id_array = np.zeros((7, 7), dtype=np.int32)
# experiment_id_array[np.triu_indices(7)] = np.arange(358, 386)
# experiment_id_array[np.tril_indices(7, -1)] = experiment_id_array.T[
#     np.tril_indices(7, -1)
# ]
# auroc_value_arrays = dict(
#     all=np.zeros((7, 7)),
#     emnist_l=np.zeros((7, 7)),
# )
# succfail_value_arrays = dict(auroc=np.zeros((7, 7)), aupr_f=np.zeros((7, 7)))

# reg_types = np.array(
#     ("cosine", "logcosine", "inversecosine", "abs", "euclid", "max", "min")
# )
# titles = ("AUROC All", "AUROC EMNIST-L", "AUROC S/F", "AUPR-F")

# for i in range(7):
#     for j in range(7):
#         if exists(
#             join(
#                 data_root,
#                 f"results_classifier_oodd_succfail_{experiment_id_array[i, j]}.csv",
#             )
#         ):
#             df = pd.read_csv(
#                 join(
#                     data_root,
#                     f"results_classifier_ood_succfail_{experiment_id_array[i, j]}.csv",
#                 ),
#                 sep=",",
#                 index_col=0,
#             )

#             for k in auroc_value_arrays.keys():
#                 auroc_value_arrays[k][i, j] = df.loc[k].loc["auroc"]

#             succfail_value_arrays["auroc"][i, j] = df.loc["succ/fail"].loc["auroc"]
#             succfail_value_arrays["aupr_f"][i, j] = df.loc["succ/fail"].loc["aupr-out"]
#         else:
#             for k in auroc_value_arrays.keys():
#                 auroc_value_arrays[k][i, j] = np.nan

#             succfail_value_arrays["auroc"][i, j] = np.nan
#             succfail_value_arrays["aupr_f"][i, j] = np.nan

# auroc_value_arrays.update(succfail_value_arrays)

# normalized_value_arrays = dict()
# for k in auroc_value_arrays.keys():
#     mask = ~np.isnan(auroc_value_arrays[k])
#     normalized_value_arrays[k] = (
#         auroc_value_arrays[k] - auroc_value_arrays[k][mask].min()
#     ) / (auroc_value_arrays[k][mask].max() - auroc_value_arrays[k][mask].min())

# ax: Tuple[plt.Axes]
# fig, ax = plt.subplots(2, 2, figsize=(13, 14))
# ax = ax.reshape(-1)

# # -----
# for i, (k, v) in enumerate(normalized_value_arrays.items()):
#     ax[i].imshow(v.T, interpolation="nearest", cmap=cm.coolwarm)

#     # Taken from: https://stackoverflow.com/a/22160419
#     x, y = np.meshgrid(np.arange(len(reg_types)), np.arange(len(reg_types)))
#     for xval, yval in zip(x.flatten(), y.flatten()):
#         ax[i].text(
#             xval,
#             yval,
#             f"{auroc_value_arrays[k][xval, yval]:.2%}",
#             va="center",
#             ha="center",
#         )

#     ax[i].set_xticks(np.arange(len(reg_types)) + 0.5)
#     ax[i].set_yticks(np.arange(len(reg_types)) + 0.5)
#     ax[i].grid(ls="-", lw=2)
#     ax[i].set_title(titles[i])

#     for a, ind, labels in zip(
#         (ax[i].xaxis, ax[i].yaxis),
#         (np.arange(len(reg_types)), np.arange(len(reg_types))),
#         (reg_types, reg_types),
#     ):
#         a.set_major_formatter(ticker.NullFormatter())
#         a.set_minor_locator(ticker.FixedLocator(ind))
#         a.set_minor_formatter(ticker.FixedFormatter(labels))

#     # ax[i].tick_params(axis="x", rotation=45)
#     for tick in ax[i].get_xminorticklabels():
#         tick.set_rotation(45)

# plt.savefig(join(plot_root, "reg_type_parameter_study.pdf"), bbox_inches="tight")


########################################################################################
