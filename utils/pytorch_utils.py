"""
PyTorch training utilities for optimization, scheduling, and logging.

Provides helper functions for:
- Creating optimizers (Adam, AdamW, SGD) from configurations
- Building learning rate schedulers (plateau, polynomial, cosine, warmup variants)
- Weight decay computation (L1/L2)
- TensorBoard logging setup
- Training state management
- Early stopping

Example:
    >>> optim = get_optimizer(conf.optimizer, model)
    >>> sched, sched_on_epoch = get_scheduler(conf.optimizer, optim, epochs=100)
    >>> early_stop = EarlyStopping(patience=10, mode='max')
"""

import sys
from typing import Any, Literal, Optional, Tuple, Union

import torch
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.utilities import AttributeDict

###################################################################################################


def get_optimizer(conf: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        conf: Configuration object with attributes:
            - name: Optimizer name ('adam', 'adamw', 'sgd')
            - lr: Learning rate
            - wd: Weight decay (for AdamW)
        model: PyTorch model to optimize

    Returns:
        Configured optimizer instance

    Raises:
        NotImplementedError: If optimizer name is not supported

    Example:
        >>> conf = OmegaConf.create({'name': 'adamw', 'lr': 1e-4, 'wd': 0.01})
        >>> optim = get_optimizer(conf, model)
    """
    if conf.name.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.name.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    elif conf.name.lower() == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=conf.lr)
    else:
        raise NotImplementedError
    return optim


def get_scheduler(
    conf: Any,
    optim: torch.optim.Optimizer,
    epochs: Optional[int] = None,
    mode: str = "min",
    warm_factor: float = 0.005,
    plateau_factor: float = 0.2,
) -> Tuple[torch.optim.lr_scheduler.LRScheduler, bool]:
    """
    Create learning rate scheduler from configuration.

    Supports various scheduling strategies including warmup, polynomial decay,
    cosine annealing, and plateau-based reduction.

    Args:
        conf: Configuration object with 'sched' attribute specifying strategy
        optim: Optimizer to schedule
        epochs: Total number of training epochs (required for some schedulers)
        mode: Mode for plateau scheduler ('min' or 'max')
        warm_factor: Initial learning rate factor for warmup (default: 0.005)
        plateau_factor: Factor for plateau reduction (default: 0.2)

    Returns:
        Tuple of (scheduler, sched_on_epoch) where:
            - scheduler: Configured LR scheduler
            - sched_on_epoch: Boolean indicating if scheduler steps per epoch

    Supported Strategies:
        - 'flat': Constant learning rate
        - 'plateau_<patience>': Reduce on plateau with specified patience
        - 'poly_<power>': Polynomial decay with specified power
        - 'warmpoly_<nwarm>_<power>': Warmup then polynomial decay
        - 'cosine': Cosine annealing
        - 'warmcosine_<nwarm>': Warmup then cosine annealing
        - 'sd_<ndec>': Sudden decay in last epochs
        - 'wsd_<nwarm>_<ndec>': Warmup, stable, then sudden decay

    Example:
        >>> conf = OmegaConf.create({'sched': 'warmcosine_5'})
        >>> sched, on_epoch = get_scheduler(conf, optim, epochs=100)
    """
    name = conf.sched.lower() if conf.sched is not None else "flat"
    sched_on_epoch = True
    if name == "flat":
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lambda epoch: 1.0,
        )
    elif name.startswith("plateau"):
        _, patience = name.split("_")
        patience = max(0, int(patience) - 1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode=mode,
            factor=plateau_factor,
            patience=patience,
        )
    elif name.startswith("poly"):
        _, power = name.split("_")
        power = float(power)
        sched = torch.optim.lr_scheduler.PolynomialLR(
            optim, total_iters=epochs, power=power
        )
    elif name.startswith("warmpoly"):
        _, nwarm, power = name.split("_")
        nwarm = max(1, int(nwarm))
        power = float(power)
        assert epochs > nwarm
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.PolynomialLR(
            optim, total_iters=epochs - nwarm, power=power
        )
        sched = torch.optim.lr_scheduler.SequentialLR(optim, [s1, s2], [nwarm])
    elif name.startswith("sd"):
        _, ndec = name.split("_")
        ndec = max(1, int(ndec)) + 1
        assert epochs > ndec
        s1 = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=epochs - ndec
        )
        s2 = torch.optim.lr_scheduler.PolynomialLR(optim, power=2, total_iters=ndec)
        sched = torch.optim.lr_scheduler.SequentialLR(optim, [s1, s2], [epochs - ndec])
    elif name.startswith("wsd"):
        _, nwarm, ndec = name.split("_")
        nwarm = max(1, int(nwarm))
        ndec = max(1, int(ndec)) + 1
        assert epochs > nwarm + ndec
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=epochs - nwarm - ndec
        )
        s3 = torch.optim.lr_scheduler.PolynomialLR(optim, power=2, total_iters=ndec)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, [s1, s2, s3], [nwarm, epochs - ndec]
        )
    elif name == "cosine":
        # pure cosine annealing over all epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=epochs,
            eta_min=0.0,
        )
    elif name.startswith("warmcosine"):
        # syntax: "warmcosine_<nwarm>"
        _, nwarm = name.split("_")
        nwarm = max(1, int(nwarm))
        assert epochs > nwarm, "Total epochs must exceed warm-up epochs"
        s1 = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=warm_factor, end_factor=1.0, total_iters=nwarm
        )
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=epochs - nwarm,
            eta_min=0.0,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=[s1, s2],
            milestones=[nwarm],
        )
    else:
        raise NotImplementedError
    return sched, sched_on_epoch


###################################################################################################


def weight_decay(
    model: torch.nn.Module,
    lamb: float,
    optim_name: str,
    form: Literal["l1", "l2"] = "l1",
    excluded_optimizers: Tuple[str, ...] = ("adamw", "soap"),
    considered_layers: Tuple[type, ...] = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
    ),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weight decay regularization for model parameters.

    Calculates L1 or L2 norm of weights from specified layer types, excluding
    optimizers that already handle weight decay internally (e.g., AdamW).

    Args:
        model: PyTorch model to compute weight decay for
        lamb: Weight decay coefficient (lambda)
        optim_name: Optimizer name (lowercase, e.g., 'adamw', 'adam')
        form: Regularization form ('l1' or 'l2')
        excluded_optimizers: Optimizer names that handle decay internally
        considered_layers: Layer types to include in decay computation

    Returns:
        Tuple of (weighted_decay, raw_decay) where:
            - weighted_decay: lamb * raw_decay (for loss addition)
            - raw_decay: Unweighted decay value (for logging)

    Example:
        >>> wd_loss, wd_value = weight_decay(model, lamb=0.01, optim_name='adam')
        >>> total_loss = main_loss + wd_loss
    """
    assert form in ("l1", "l2")
    if optim_name in excluded_optimizers:
        lamb = 0
    num = torch.zeros(1, device=model.device)
    den = 0
    for m in model.modules():
        if isinstance(m, considered_layers):
            w = m.weight
            n = m.weight.numel()
            if form == "l1":
                w = w.abs()
            elif form == "l2":
                w = w.pow(2)
            num += w.sum()
            den += n
    wd = num / den
    return lamb * wd, wd


###################################################################################################


def get_logger(path: str) -> TensorBoardLogger:
    """
    Create TensorBoard logger for training metrics.

    Args:
        path: Root directory for TensorBoard logs

    Returns:
        Configured TensorBoardLogger instance

    Example:
        >>> logger = get_logger('logs/my_experiment')
    """
    return TensorBoardLogger(
        root_dir=path,
        name="",
        version="",
        default_hp_metric=False,
    )


###################################################################################################


def set_state(state: AttributeDict) -> Tuple[Any, ...]:
    """
    Extract training state components from AttributeDict.

    Args:
        state: AttributeDict containing training state

    Returns:
        Tuple of (model, optim, sched, conf, epoch, lr, cost_best)
    """
    return (
        state.model,
        state.optim,
        state.sched,
        state.conf,
        state.epoch,
        state.lr,
        state.cost_best,
    )


def get_state(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    conf: Any,
    epoch: int,
    lr: float,
    cost_best: float,
) -> AttributeDict:
    """
    Package training state into AttributeDict for checkpointing.

    Args:
        model: PyTorch model
        optim: Optimizer
        sched: LR scheduler
        conf: Configuration object
        epoch: Current epoch number
        lr: Current learning rate
        cost_best: Best validation cost achieved

    Returns:
        AttributeDict containing all training state components

    Example:
        >>> state = get_state(model, optim, sched, conf, epoch=10, lr=1e-4, cost_best=0.5)
        >>> torch.save(state, 'checkpoint.pt')
    """
    return AttributeDict(
        model=model,
        optim=optim,
        sched=sched,
        conf=conf,
        epoch=epoch,
        lr=lr,
        cost_best=cost_best,
    )


###################################################################################################


class LogDict:
    """
    Dictionary for accumulating and synchronizing training metrics.

    Stores metrics across batches and supports distributed synchronization
    via Lightning Fabric. Handles automatic CPU transfer and concatenation.

    Example:
        >>> log = LogDict()
        >>> log.append({'loss': torch.tensor(0.5), 'acc': torch.tensor(0.9)})
        >>> log.append({'loss': torch.tensor(0.4), 'acc': torch.tensor(0.92)})
        >>> log.sync_and_mean(fabric)  # Synchronize across GPUs
        >>> metrics = log.get(['loss', 'acc'])
    """

    def __init__(self, d: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Initialize LogDict, optionally with initial data."""
        self.reset()
        if d is not None:
            self.append(d)

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self.d = {}

    def get(
        self,
        keys: Optional[Union[str, List[str]]] = None,
        prefix: str = "",
        suffix: str = "",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Retrieve metrics with optional key transformation.

        Args:
            keys: Single key (str) or list of keys. If None, returns all.
            prefix: String to prepend to all keys
            suffix: String to append to all keys

        Returns:
            Single tensor if keys is str, dict if keys is list or None
        """
        if keys is None:
            keys = list(self.d.keys())
        elif type(keys) != list:
            return self.d[keys]
        d = {}
        for key in keys:
            new_key = prefix + key + suffix
            d[new_key] = self.d[key]
        return d

    def append(self, newd: Dict[str, torch.Tensor]) -> None:
        """
        Append new metrics to accumulation.

        Args:
            newd: Dictionary of metric name -> tensor value
        """
        assert type(newd) == dict
        for key, value in newd.items():
            value = value.cpu()
            if value.ndim == 0:
                value = torch.FloatTensor([value])
            if key not in self.d:
                self.d[key] = value
            else:
                self.d[key] = torch.cat([self.d[key], value], dim=0)

    def sync_and_mean(self, fabric: Any) -> None:
        """
        Synchronize metrics across all GPUs and compute mean.

        Args:
            fabric: Lightning Fabric instance for distributed operations

        Side Effects:
            Replaces all stored tensors with scalar mean values
        """
        fabric.barrier()
        for key in self.d.keys():
            self.d[key] = fabric.all_gather(self.d[key]).mean().item()


#########################################################################################
# Early Stopping Class
#########################################################################################


class EarlyStopping:
    """
    Early stopping utility to halt training when validation metric plateaus.

    Monitors validation metric and stops training after specified patience
    (number of epochs) without improvement. Supports both minimization
    (loss) and maximization (accuracy, MAP) objectives.

    Attributes:
        patience: Number of epochs to wait for improvement
        mode: 'min' for loss, 'max' for accuracy/MAP
        min_delta: Minimum change to qualify as improvement
        enabled: Whether early stopping is active
        best_score: Best metric value seen so far
        counter: Epochs since last improvement
        early_stop: Flag indicating if stopping criterion is met

    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max', min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_map = validate()
        ...     if early_stop(val_map):
        ...         print(f"Stopping at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        mode: Literal["min", "max"] = "max",
        min_delta: float = 0.0,
        enabled: bool = True,
    ) -> None:
        """
        Initialize early stopping monitor.

        Args:
            patience: Number of epochs to wait for improvement (default: 10)
            mode: 'min' for decreasing metrics (loss), 'max' for increasing
                 metrics (accuracy/MAP). Default: 'max'
            min_delta: Minimum change to qualify as improvement (default: 0.0)
            enabled: Whether early stopping is enabled (default: True)
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.enabled = enabled
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop based on current validation score.

        Args:
            score: Current validation metric value

        Returns:
            True if training should stop (patience exhausted), False otherwise

        Example:
            >>> early_stop = EarlyStopping(patience=3, mode='max')
            >>> early_stop(0.85)  # First epoch
            False
            >>> early_stop(0.86)  # Improved
            False
            >>> early_stop(0.86)  # No improvement (1/3)
            False
            >>> early_stop(0.85)  # No improvement (2/3)
            False
            >>> early_stop(0.84)  # No improvement (3/3) - stop!
            True
        """
        if not self.enabled:
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        improved = False
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


###################################################################################################
