# Standard Library
import logging
from typing import Union

# Thirdparty libraries
from sklearn.metrics import roc_auc_score as rocauc
import torch

# see
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html


def auroc(
    pred: torch.Tensor, target: torch.Tensor, autonorm: bool = False
) -> Union[None, float]:
    """Return area under receiver operating curve

    Args:
        pred (torch.Tensor): Prediction scores, should take values in [0, 1]
            if autonorm is False.
        target (torch.Tensor): Ground truth labels, should take values in {0, 1}
        autonorm (bool, optional): If true, predictions will be normalized to the
            interval [0, 1]. Defaults to False.

    Raises:
        ValueError: If autonorm is False and the prediction scores are not in the
            interval [0, 1]

    Returns:
        Union[None, float]: AUROC or None if the prediction scores contain only a
            single value (min=max)
    """

    log = logging.getLogger("eval.binary.auroc")

    log.debug(f"'pred' value range: [{pred.min():.3f}, {pred.max():.3f}]")

    pred = pred.view(-1)
    target = target.view(-1)

    if autonorm:
        pred_max = pred.max()
        pred_min = pred.min()

        if pred_max != pred_min:
            pred = (pred - pred_min) / (pred_max - pred_min)
    elif pred.min() < 0 or pred.max() > 1:
        raise ValueError("Prediction values should be in the interval [0,1]")

    if pred.max() != pred.min():
        return rocauc(target.numpy(), pred.numpy())
    else:
        log.warning("'pred' only contains a single value, returning None!")
        return None
