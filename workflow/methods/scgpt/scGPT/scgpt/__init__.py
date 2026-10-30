__version__ = "0.2.5"
import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- ExPert-benchmarks patch ------------------------------------------------
# The benchmark only uses scgpt.model / tokenizer / loss / utils, which depend
# on nothing heavier than torch/anndata/numpy/pandas/tqdm. Upstream's __init__
# also eagerly imports scbank/tasks/trainer/data_*, which pull in datasets,
# scanpy, wandb, scib, … — deps we deliberately keep out of the lean env. Keep
# the needed submodules eager and make the rest best-effort so `import scgpt`
# works without them. Restore the unconditional upstream imports if you need the
# full feature set.
from . import model, tokenizer, utils

try:
    from . import scbank, tasks
    from .data_collator import DataCollator
    from .data_sampler import SubsetsBatchSampler
    from .trainer import (
        prepare_data,
        prepare_dataloader,
        train,
        define_wandb_metrcis,
        evaluate,
        eval_testdata,
        test,
    )
except ImportError as _exc:  # optional heavy deps absent in the lean env
    logger.debug("scGPT optional submodules unavailable (%s)", _exc)
# --- end patch --------------------------------------------------------------
