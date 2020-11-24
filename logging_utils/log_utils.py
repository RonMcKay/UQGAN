import logging
from typing import Optional

from sacred.commands import _format_config
from sacred.run import Run
import torch


def log_config(_run: Run, _log: logging.Logger):
    """Prints the sacred experiment configuration with the same formatting as in the
    'print_config' method.

    Args:
        _run (Run): A sacred _run object.
        _log (Logger): Logger from the sacred experiment.
    """
    final_config = _run.config
    config_mods = _run.config_modifications
    _log.info(_format_config(final_config, config_mods))


def log_cuda_memory(_run: Optional[Run] = None):
    log = logging.getLogger("cuda-memory-usage")
    current_cuda_memory = torch.cuda.memory_allocated()
    max_cuda_memory = torch.cuda.max_memory_allocated()
    log.debug("current: {}, max: {}".format(current_cuda_memory, max_cuda_memory))
    if _run is not None:
        _run.log_scalar("cuda_mem", current_cuda_memory)
        _run.log_scalar("cuda_max_mem", max_cuda_memory)
