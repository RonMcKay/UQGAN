# Standard Library
import logging

# Thirdparty libraries
import torch


def fprxtpr(
    pred: torch.Tensor,
    target: torch.Tensor,
    tpr_threshold: float = 0.95,
    autonorm: bool = False,
    return_threshold: bool = False,
) -> float:
    """Compute the FPR at <tpr_threshold> TPR

    Args:
        pred (torch.Tensor): Prediction scores, should take values in [0, 1]
            if autonorm is False.
        target (torch.Tensor): Ground truth labels, should take values in {0, 1}.
        tpr_threshold (float, optional): The TPR at which to compute the FPR. Has to be
            in the interval [0, 1]. Defaults to 0.95.
        autonorm (bool, optional): If true, predictions will be normalized to the
            interval [0, 1]. Defaults to False.
        return_threshold (bool, optional): If true, also returns a relative
            (relative to min/max of pred) threshold at which <tpr_threshold> TPR is
            reached. Defaults to False.

    Raises:
        ValueError: If tpr_threshold is not in the interval [0, 1]
        ValueError: If autonorm is False and prediction scores are not in the
            interval [0, 1]
        ValueError: If any element of target is not in {0, 1}

    Returns:
        float: FPR at <tpr_threshold> TPR
    """
    log = logging.getLogger("eval.binary.fprxtpr")

    if not 0 <= tpr_threshold <= 1:
        raise ValueError("'tpr_threshold' should be in the interval [0,1]")

    log.debug(f"'pred' value range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")

    pred = pred.view(-1)
    target = target.view(-1)

    if autonorm:
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    elif pred.min() < 0 or pred.max() > 1:
        raise ValueError(
            "Prediction values should be in the interval [0, 1] if autonorm is False"
        )

    if not torch.all(torch.unique(target) == torch.Tensor([0, 1])):
        raise ValueError("'target' only needs to contain one of {0, 1}")

    pred_sorted_inds = torch.argsort(pred)

    # find threshold
    positive_instances = 0
    for i in range(target.shape[0]):
        if target[pred_sorted_inds][-i] == 1:
            positive_instances += 1
        if positive_instances / target.sum() >= tpr_threshold:
            break
    threshold = pred[pred_sorted_inds][-i]

    fpr = (target[pred >= threshold] == 0).long().sum() / (target == 0).long().sum()

    if return_threshold:
        return fpr.item(), threshold
    else:
        return fpr.item()


if __name__ == "__main__":
    # Thirdparty libraries
    import matplotlib.pyplot as plt

    x1 = torch.randn((10000))
    x2 = torch.randn((10000)) + 3.5
    x = torch.cat((x1, x2))

    y1 = torch.zeros((x1.shape[0]))
    y2 = torch.ones((x2.shape[0]))
    y = torch.cat((y1, y2))

    fig, ax = plt.subplots(1)
    ax.hist(x1.numpy(), 50, facecolor="r")
    ax.hist(x2.numpy(), 50, facecolor="g", alpha=0.6)
    out, threshold = fprxtpr(x, y, autonorm=True, return_threshold=True)
    ax.axvline(x.min() + (x.max() - x.min()) * threshold, color="b")

    plt.savefig("debug_plot.jpg")

    print(out)
