from typing import Any

from torch import Tensor

from .base import BaseCAE


class IdentityCAE(BaseCAE):
    def __init__(self, cl_dim: int = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cl_dim = cl_dim

    def _encode(self, x: Tensor, cl: Tensor) -> Tensor:
        return x

    def _decode(self, encoding: Tensor, cl: Tensor) -> Tensor:
        return encoding
