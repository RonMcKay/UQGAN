# Taken from this PR: https://github.com/Lightning-AI/lightning/pull/656
"""
Log using `sacred <https://sacred.readthedocs.io/en/stable/index.html>'_
.. code-block:: python
    from pytorch_lightning.logging import SacredLogger
    ex = Experiment() # initialize however you like
    ex.main(your_main_fct)
    ex.observers.append(
        # add any observer you like
    )
    sacred_logger = SacredLogger(ex)
    trainer = Trainer(logger=sacred_logger)
Use the logger anywhere in you LightningModule as follows:
.. code-block:: python
    def train_step(...):
        # example
        self.logger.experiment.whatever_sacred_supports(...)
    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_sacred_supports(...)
"""

import argparse
from logging import getLogger
from typing import Dict, Optional, Union

try:
    from sacred.experiment import Experiment
    from sacred.observers import FileStorageObserver
except ImportError:
    raise ImportError("Missing sacred package.  Run `pip install sacred`")

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = getLogger(__name__)


class SacredLogger(Logger):
    def __init__(self, sacred_experiment: Experiment):
        """Initialize a sacred logger.

        Args:
            sacred_experiment:Experiment object with desired observers already appended.
        """
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name: str = sacred_experiment.path
        self._run_id = None

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        if self._run_id is None:
            self._run_id = self.sacred_experiment.current_run._id

        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        # probably not needed bc. it is dealt with by sacred
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(f"Discarding metric with string value {k}={v}")
                continue
            self.experiment.log_scalar(k, v, step)

    @property
    def save_dir(self) -> Optional[str]:
        for obs in self.experiment.observers:
            if isinstance(obs, FileStorageObserver):
                return obs.basedir
        return None

    @property
    def name(self) -> str:
        return self.experiment_name

    @property
    def version(self) -> Union[int, str]:
        return self.run_id
