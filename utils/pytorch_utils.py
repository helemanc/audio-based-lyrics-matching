import sys
import torch
from lightning.fabric.utilities import AttributeDict
from lightning.fabric.loggers import TensorBoardLogger

###################################################################################################


def get_optimizer(conf, model):
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
    conf,
    optim,
    epochs=None,
    mode="min",
    warm_factor=0.005,
    plateau_factor=0.2,
):
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
    model,
    lamb,
    optim_name,
    form="l1",
    excluded_optimizers=("adamw", "soap"),
    considered_layers=(
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
    ),
):
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


def get_logger(path):
    return TensorBoardLogger(
        root_dir=path,
        name="",
        version="",
        default_hp_metric=False,
    )


###################################################################################################


def set_state(state):
    return (
        state.model,
        state.optim,
        state.sched,
        state.conf,
        state.epoch,
        state.lr,
        state.cost_best,
    )


def get_state(model, optim, sched, conf, epoch, lr, cost_best):
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

    def __init__(self, d=None):
        self.reset()
        if d is not None:
            self.append(d)

    def reset(self):
        self.d = {}

    def get(self, keys=None, prefix="", suffix=""):
        if keys is None:
            keys = list(self.d.keys())
        elif type(keys) != list:
            return self.d[keys]
        d = {}
        for key in keys:
            new_key = prefix + key + suffix
            d[new_key] = self.d[key]
        return d

    def append(self, newd):
        assert type(newd) == dict
        for key, value in newd.items():
            value = value.cpu()
            if value.ndim == 0:
                value = torch.FloatTensor([value])
            if key not in self.d:
                self.d[key] = value
            else:
                self.d[key] = torch.cat([self.d[key], value], dim=0)

    def sync_and_mean(self, fabric):
        fabric.barrier()
        for key in self.d.keys():
            self.d[key] = fabric.all_gather(self.d[key]).mean().item()

#########################################################################################
# Early Stopping Class
#########################################################################################

class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience=10, mode='max', min_delta=0.0, enabled=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement
            mode (str): 'min' for decreasing metrics (loss), 'max' for increasing (accuracy/MAP)
            min_delta (float): Minimum change to qualify as improvement
            enabled (bool): Whether early stopping is enabled
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.enabled = enabled
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score):
        """
        Call this method with the current validation metric
        Returns True if training should stop
        """
        if not self.enabled:
            return False
            
        if self.best_score is None:
            self.best_score = score
            return False
            
        improved = False
        if self.mode == 'min':
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

def set_state(state):
    return (
        state.model,
        state.optim,
        state.sched,
        state.conf,
        state.epoch,
        state.lr,
        state.cost_best,
    )


def get_state(model, optim, sched, conf, epoch, lr, cost_best):
    return AttributeDict(
        model=model,
        optim=optim,
        sched=sched,
        conf=conf,
        epoch=epoch,
        lr=lr,
        cost_best=cost_best,
    )
