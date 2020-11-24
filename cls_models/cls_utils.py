import torch.nn as nn
from torch.nn.modules.dropout import _DropoutNd


def _set_dropout_to_train_mode(m: nn.Module) -> None:
    if isinstance(m, _DropoutNd):
        m.train()


def set_model_to_mode(classifier: nn.Module, mode: str) -> None:
    if mode == "train":
        classifier.train()
    elif mode == "eval":
        classifier.eval()
        # Only set dropout to training mode (for test time dropout)
        if getattr(classifier, "mc_dropout", 0.0) > 0:
            classifier.apply(_set_dropout_to_train_mode)
    else:
        raise ValueError(f"Unknown mode '{mode}'")
