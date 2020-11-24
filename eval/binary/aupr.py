# Standard Library
import logging
from typing import Union

# Thirdparty libraries
from sklearn.metrics import auc, precision_recall_curve
import torch

# see
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html


def aupr(
    pred: torch.Tensor, target: torch.Tensor, autonorm=False
) -> Union[None, float]:
    """Return area under precision recall curve

    Args:
        pred (torch.Tensor): Prediction scores, should take values in [0, 1] if
            autonorm is False.
        target (torch.Tensor): Ground truth labels, should take values in {0, 1}
        autonorm (bool, optional): If true, predictions will be normalized to the
            interval [0, 1]. Defaults to False.

    Raises:
        ValueError: If autonorm is False and the prediction scores are not in the
            interval [0, 1]

    Returns:
        Union[None, float]: AUPR or None if the prediction scores contain only a single
            value (min=max)
    """

    log = logging.getLogger("eval.binary.aupr")

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

    # get the precision and recall values
    precision, recall, thresholds = precision_recall_curve(target, pred)

    precision = torch.from_numpy(precision).view(-1)
    recall = torch.from_numpy(recall).view(-1)

    # order recall for the auc computation
    perm = torch.argsort(recall)
    recall = recall[perm]
    precision = precision[perm]

    if pred.max() != pred.min():
        # compute area under the precision recall curve
        return auc(recall, precision)
    else:
        log.warning("'pred' only contains a single value, returning None!")
        return None
