import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from scipy.stats import gaussian_kde
import torch

from cls_models import ToyClassifier, cls_models, load_cls_model
from datasets import (  # noqa
    ToyDataset2,
    ToyDataset3,
    ToyDataset4,
    ToyDataset5,
    datasets,
)
from gan_models import ToyDiscriminator, ToyGenerator, gan_models, load_gan_model
from logging_utils import log_config
from utils import init_experiment

ex = Experiment("Plot Toy Example", ingredients=[gan_models, cls_models, datasets])
init_experiment(ex, mongo_observer=False)


def interpolate_colormap(color_1, color_2, n=1000) -> colors.Colormap:
    t = torch.arange(0, 1 + 1.0 / n, 1.0 / n).view(-1, 1)
    cols = color_1.flatten().unsqueeze(0).repeat(n + 1, 1) * (
        1 - t
    ) + t * color_2.flatten().unsqueeze(0).repeat(n + 1, 1)
    col_map = colors.ListedColormap(cols.numpy())
    return col_map


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000) -> colors.Colormap:
    # taken from https://stackoverflow.com/a/18926541/9339669
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


@cls_models.config
def classifier_config_update(cfg):
    cfg["name"] = "toy"
    cfg["checkpoint"] = "<classifier checkpoint path>"


@gan_models.config
def gan_config_update(cfg):
    cfg["name"] = "toy"
    cfg["gen_checkpoint"] = "<generator checkpoint path>"
    cfg["disc_checkpoint"] = "<discriminator checkpoint path>"


@ex.automain
def main(gan_model, cls_model, _run, _log):
    log_config(_run, _log)
    gpu = 0
    batch_size = 64
    max_visual_points_real = 300
    max_visual_points_gan = 300
    dataset_size = 100000
    heatmap_cm_factor = 0.65
    heatmap_resolution = 500
    xymin, xymax = None, None
    # default_colors = ["black", "dodgerblue"]

    if torch.cuda.is_available() and gpu is not None:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")

    # dat = ToyDataset2(n_samples=dataset_size)
    dat = ToyDataset4(n_samples=dataset_size)

    cm = plt.get_cmap("Set1")
    default_colors = [cm(i / float(dat.CL_DIM)) for i in range(dat.CL_DIM)]

    generator: ToyGenerator
    discriminator: ToyDiscriminator
    classifier: ToyClassifier

    generator, discriminator = load_gan_model()
    classifier = load_cls_model()

    generator, discriminator, classifier = (
        generator.to(device),
        discriminator.to(device),
        classifier.to(device),
    )

    in_distribution_data = dat.data
    y_onehot = dat.y_onehot

    in_dist_colors = []
    for i, count in enumerate(np.unique(y_onehot, axis=0, return_counts=True)[1]):
        in_dist_colors += [colors.to_rgb(default_colors[i])] * count

    in_dist_colors = np.stack(in_dist_colors)

    perm = torch.randperm(dat.data.shape[0])
    in_distribution_data = in_distribution_data[perm]
    y_onehot = y_onehot[perm]
    in_dist_colors = in_dist_colors[perm]

    with torch.no_grad():
        out_of_distribution_data = torch.empty((0,))
        for i, col in enumerate(y_onehot.split(split_size=batch_size)):
            col = col.to(device)
            out_of_distribution_data = torch.cat(
                (out_of_distribution_data, generator(col).cpu())
            )

    if xymin is None or xymax is None:
        scale = 1.5 if isinstance(dat, ToyDataset3) else 1.2

        xmin = in_distribution_data[:, 0].min().numpy()
        xmax = in_distribution_data[:, 0].max().numpy()
        ymin = in_distribution_data[:, 1].min().numpy()
        ymax = in_distribution_data[:, 1].max().numpy()

        x_diff = abs(xmax - xmin)
        y_diff = abs(ymax - ymin)
        xmin = xmin - x_diff * max(scale - 1, 0)
        xmax = xmax + x_diff * max(scale - 1, 0)
        ymin = ymin - y_diff * max(scale - 1, 0)
        ymax = ymax + y_diff * max(scale - 1, 0)

        # xymin = min(
        #     xmin - abs(xmax - xmin) * max(scale - 1, 0),
        #     ymin - abs(ymax - ymin) * max(scale - 1, 0),
        # )
        # xymax = max(
        #     xmax + abs(xmax - xmin) * max(scale - 1, 0),
        #     ymax + abs(ymax - ymin) * max(scale - 1, 0),
        # )

    fig_ax = []
    for _ in range(3):
        fig_ax.append(plt.subplots(nrows=1, ncols=1, figsize=(3.25063, 2.0)))

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    for fig, ax in fig_ax:
        ax.axis("off")

        ax.set_xlim(left=xmin, right=xmax)
        ax.set_ylim(bottom=ymin, top=ymax)

        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.scatter(
            in_distribution_data[:max_visual_points_real, 0],
            in_distribution_data[:max_visual_points_real, 1],
            c=in_dist_colors[:max_visual_points_real],
            marker="x",
            alpha=1.0,
            s=6,
        )

        ax.scatter(
            out_of_distribution_data[:max_visual_points_gan, 0],
            out_of_distribution_data[:max_visual_points_gan, 1],
            c=in_dist_colors[:max_visual_points_gan],
            # edgecolor="orchid",
            linewidth=0.0,
            marker="^",
            alpha=0.4,
            s=10,
        )

        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width=xmax - xmin,
                height=ymax - ymin,
                fill=False,
                linewidth=1,
                zorder=20,
            )
        )

    # build grid
    grid_x, grid_y = np.mgrid[
        xmin : xmax : complex(heatmap_resolution),
        ymin : ymax : complex(heatmap_resolution),
    ]  # type: ignore
    heatmap_points = torch.cat(
        (
            torch.tensor(grid_x.flatten()).unsqueeze(-1),
            torch.tensor(grid_y.flatten()).unsqueeze(-1),
        ),
        dim=1,
    ).float()

    ####################################################################################

    # # Plot discriminator decision boundary
    # _log.info("Computing discriminator decision boundary...")
    # heatmap_onehot = ToyDataset2.construct_onehot(heatmap_points)
    # heatmap_values_disc = torch.empty((0,))
    # with torch.no_grad():  # type: ignore
    #     for x, y_oh in zip(
    #         heatmap_points.split(split_size=256, dim=0),  # type: ignore
    #         heatmap_onehot.split(split_size=256, dim=0),  # type: ignore
    #     ):
    #         x, y_oh = x.to(device), y_oh.to(device)
    #         heatmap_values_disc = torch.cat(
    #             (heatmap_values_disc, discriminator(x, y_oh).cpu())
    #         )

    # _log.debug(
    #     (
    #         f"'heatmap_values_disc value range: [{heatmap_values_disc.min()}, "
    #         f"{heatmap_values_disc.max()}]'"
    #     )
    # )
    # heatmap_values_disc = (heatmap_values_disc - heatmap_values_disc.min()) / (
    #     heatmap_values_disc.max() - heatmap_values_disc.min()
    # )

    # axes[0, 0].set_title("discriminator decision boundary")
    # axes[0, 0].pcolormesh(
    #     grid_x,
    #     grid_y,
    #     heatmap_values_disc.numpy().reshape(grid_x.shape),
    #     cmap=plt.get_cmap("RdYlGn"),
    #     shading="gouraud",
    #     zorder=0,
    # )

    ####################################################################################

    # Plot classifier outputs
    _log.info("Computing classifier heatmaps")
    heatmap_eu = torch.empty((0,))
    heatmap_au = torch.empty((0,))
    with torch.no_grad():  # type: ignore
        for x in heatmap_points.split(split_size=256, dim=0):  # type: ignore
            x = x.to(device)

            tmp_out = classifier(x)
            heatmap_au = torch.cat((heatmap_au, tmp_out[1]))
            heatmap_eu = torch.cat((heatmap_eu, tmp_out[2]))

    # heatmap_eu = 1 - heatmap_eu
    heatmap_eu = (heatmap_eu - heatmap_eu.min()) / (heatmap_eu.max() - heatmap_eu.min())
    _log.debug(
        (f"'heatmap_in_dist value range: [{heatmap_eu.min()}, " f"{heatmap_eu.max()}]'")
    )

    # heatmap_au -= heatmap_au.min()
    heatmap_au = (heatmap_au - heatmap_au.min()) / (heatmap_au.max() - heatmap_au.min())
    _log.debug(
        (
            "'heatmap_aleatoric_uncertainty value range: "
            f"[{heatmap_au.min()}, "
            f"{heatmap_au.max()}]'"
        )
    )

    my_cmap = plt.cm.RdYlGn(np.arange(plt.cm.RdYlGn.N))
    my_cmap[:, 0:3] *= heatmap_cm_factor
    my_cmap[:, 0:3] += (1 - heatmap_cm_factor) * np.ones_like(my_cmap[:, 0:3])
    my_cmap = colors.ListedColormap(my_cmap)

    # fig_ax[0][1].set_title("Classifier OoD Heatmap")
    fig_ax[0][1].pcolormesh(
        grid_x,
        grid_y,
        heatmap_eu.numpy().reshape(grid_x.shape),
        # cmap=truncate_colormap(plt.get_cmap('Oranges'), maxval=1.0),
        cmap=interpolate_colormap(
            torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 144.0 / 255, 59.0 / 255])
        ),
        shading="gouraud",
        zorder=0,
    )

    fig_ax[0][0].savefig(
        "./visualization/plots/toy_example_ood_heatmap.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1000,
    )

    # fig_ax[1][1].set_title("Classifier Aleatoric Uncertainty")
    fig_ax[1][1].pcolormesh(
        grid_x,
        grid_y,
        heatmap_au.numpy().reshape(grid_x.shape),
        # cmap=truncate_colormap(plt.get_cmap('Oranges'), maxval=1.0),
        cmap=interpolate_colormap(
            torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 144.0 / 255, 59.0 / 255])
        ),
        shading="gouraud",
        zorder=0,
    )

    fig_ax[1][0].savefig(
        "./visualization/plots/toy_example_au_heatmap.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1000,
    )

    ####################################################################################

    _log.info("Computing estimated density")
    # plot estimated density of generated samples
    col_arrays = []
    densities = []
    for i, col in enumerate(np.unique(in_dist_colors, axis=0)):
        kde_data = out_of_distribution_data[(in_dist_colors == col).sum(1) == 3]
        embedding_kde = gaussian_kde(kde_data.numpy().T)
        grid_density = embedding_kde(np.vstack([grid_x.flatten(), grid_y.flatten()]))
        tmp = torch.arange(0, 1.001, 0.001).view(-1, 1).expand((-1, 3))
        colmap = colors.ListedColormap(
            (
                torch.tensor(colors.to_rgb(default_colors[i])) * tmp
                + (1 - tmp) * torch.tensor([1.0, 1.0, 1.0])
            ).numpy()
        )
        grid_density = (grid_density - grid_density.min()) / (
            grid_density.max() - grid_density.min()
        )
        densities.append(grid_density.reshape(grid_x.shape).copy())
        # grid_density[grid_density < np.quantile(grid_density, 0.55)] = np.NaN
        # colmap.set_bad('white')

        col_arrays.append(colmap(grid_density.reshape(grid_x.shape)))

    density_margin_threshold = 0.3

    final_color = col_arrays[0]

    col_mask = densities[1] > (densities[0] + density_margin_threshold)
    final_color[col_mask] = col_arrays[1][col_mask]

    col_mask = np.abs(densities[0] - densities[1]) <= density_margin_threshold
    final_color[col_mask] *= 0.5
    final_color[col_mask] += 0.5 * col_arrays[1][col_mask]

    fig_ax[2][1].imshow(
        np.swapaxes(final_color, 0, 1),
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
        zorder=10,
        alpha=0.8,
        interpolation="bilinear",
        origin="lower",
    )

    # fig_ax[2][1].imshow(
    #     final_color,
    #     extent=[xymin, xymax, xymin, xymax],
    #     aspect="auto",
    #     zorder=10,
    #     alpha=0.8,
    #     interpolation="bilinear",
    #     origin="lower",
    # )

    fig_ax[2][0].savefig(
        "./visualization/plots/toy_example_density.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1000,
    )

    ####################################################################################
    # Plot correlation between real distribution and in-distribution likelihood

    # in_dist_likelihood = torch.empty((0,))
    # with torch.no_grad():
    #     for x, y_oh in zip(
    #         in_distribution_data.split(split_size=batch_size),
    #         y_onehot.split(split_size=batch_size),
    #     ):
    #         x = x.to(device)
    #         out = classifier(x).cpu()
    #         in_dist_likelihood = torch.cat(
    #             (in_dist_likelihood, torch.sigmoid(out)[y_oh == 1])
    #         )

    # real_likelihood = torch.empty((0,))
    # for i, m in enumerate(dat.means):
    #     n = dat.n_samples // len(dat.means)
    #     real_likelihood = torch.cat(
    #         (
    #             real_likelihood,
    #             torch.exp(
    #                 MultivariateNormal(m, torch.eye(2)).log_prob(
    #                     in_distribution_data[n * i : n * (i + 1)]
    #                 )
    #             ),
    #         )
    #     )

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25063, 3.25063))

    # seaborn.kdeplot(
    #     x=in_dist_likelihood, y=real_likelihood, shade=True, ax=ax, clip=(0, 1)
    # )

    # ax.scatter(
    #     in_dist_likelihood,
    #     real_likelihood,
    #     c=in_dist_colors,
    #     alpha=0.5,
    #     s=0.1,
    # )

    # fig.savefig(
    #     "./visualization/plots/toy_example_likelihood_scatter.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    #     dpi=1000,
    # )

    ####################################################################################

    plt.close()
