# This is an implementation for the Expected Calibration Error (ECE).
# See also: Guo et al. 2017 (https://arxiv.org/abs/1706.04599)

# Thirdparty libraries
import torch


def ece(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    bins: int = 15,
    autonorm: bool = False,
) -> float:
    pred = pred.flatten(start_dim=0)
    target = target.flatten(start_dim=0)
    confidence = confidence.flatten(start_dim=0)

    correct_prediction = (pred == target).float()

    if autonorm:
        confidence = (confidence - confidence.min()) / (
            confidence.max() - confidence.min()
        )
    else:
        min_conf = confidence.min()
        max_conf = confidence.max()

        if min_conf > 1.0 or min_conf < 0.0 or max_conf > 1.0 or max_conf < 0.0:
            raise ValueError(
                "If autonorm is False, 'confidence' should contain "
                "values in the interval [0, 1]"
            )

    final_ece = torch.tensor(0.0)
    for i in range(bins):
        lb = float(i) / bins
        ub = float(i + 1) / bins

        mask = torch.bitwise_and(lb < confidence, confidence <= ub)

        if mask.long().sum() == 0:
            continue

        bin_acc = correct_prediction[mask].mean()
        bin_conf = confidence[mask].mean()

        final_ece += (bin_acc - bin_conf).abs() * (
            mask.long().sum() / confidence.shape[0]
        )

    return final_ece.item()


if __name__ == "__main__":
    preds = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5)).sample((100,))
    targets = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.5)).sample((100,))
    confidences = torch.rand(100)

    print(ece(preds, targets, confidences))

    preds = torch.ones((100,))
    targets = torch.zeros((100,))
    confidences = torch.ones((100,))

    print(ece(preds, targets, confidences))

    preds = torch.ones((100,))
    targets = torch.ones((100,))
    confidences = torch.ones((100,))

    print(ece(preds, targets, confidences))
