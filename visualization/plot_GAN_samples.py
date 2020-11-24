from sacred import Experiment
import torch
import torch.nn.functional as tf
from torchvision.utils import make_grid, save_image

from cae_models import cae_models, load_cae_model
from datasets import datasets
from gan_models import gan_models, load_gan_model
from logging_utils import log_config

ex = Experiment("Plot GAN OOD samples", ingredients=[datasets, gan_models, cae_models])
device = None


@ex.config
def config():
    args = dict(  # noqa: F841
        classes=[],
        n_samples=4,
        batch_size=128,
        gpu=0,
    )


@ex.automain
def main(args, dataset, gan_model, cae_model, _run, _log):
    log_config(_run, _log)

    ########################################
    #              Set device
    ########################################

    if torch.cuda.is_available() and args["gpu"] is not None:
        device = torch.device("cuda:{}".format(args["gpu"]))
    else:
        device = torch.device("cpu")

    overwrite_gan_cfg = dict(
        name="toy",
        output_size=cae_model["cfg"]["latent_dim"],
        input_size=cae_model["cfg"]["latent_dim"],
        cl_dim=dataset["cfg"]["cl_dim"],
    )

    generator, _ = load_gan_model(**overwrite_gan_cfg)
    generator = generator.to(device)

    cae = load_cae_model()
    cae = cae.to(device)

    y_onehot = torch.empty((0,))
    for cl in args["classes"]:
        y_onehot = torch.cat(
            (
                y_onehot,
                tf.one_hot(
                    torch.tensor(cl, dtype=torch.long),
                    num_classes=dataset["cfg"]["cl_dim"],
                )
                .unsqueeze(0)
                .repeat(args["n_samples"], 1),
            )
        )

    all_samples = torch.empty((0,))
    for i in range(args["n_samples"]):
        y_oh = y_onehot.index_select(
            0,
            torch.tensor(
                [i + j * args["n_samples"] for j in range(len(args["classes"]))],
                dtype=torch.long,
            ),
        )
        y_oh = y_oh.to(device)

        x_encoding_tilde = generator(y_oh)
        x_tilde = torch.sigmoid(cae.decode(x_encoding_tilde, y_oh))
        all_samples = torch.cat((all_samples, x_tilde.cpu()))

    img_grid = make_grid(all_samples, nrow=len(args["classes"]), pad_value=1)
    save_image(img_grid, fp=f"visualization/plots/{dataset['cfg']['name']}_samples.png")
