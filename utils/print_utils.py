"""
Printing and progress bar utilities for distributed training.

Provides conditional printing functions that respect distributed training settings,
formatted metric reporting, and progress bars with customizable options.

Example:
    >>> myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    >>> myprint("Training started")  # Only prints on rank 0
    >>> for batch in myprogbar(dataloader, desc="Training"):
    ...     # Training loop
"""

import sys
import time
from typing import Any, Dict, Optional

from tqdm import tqdm

###################################################################################################


def myprint(s: str, end: str = "\n", doit: bool = True, flush: bool = True) -> None:
    """
    Conditional print function for distributed training.

    Prints message only if doit is True, useful for restricting output to
    rank 0 in distributed settings.

    Args:
        s: String to print
        end: Line ending character (default: newline)
        doit: Whether to actually print (default: True)
        flush: Whether to flush output buffer (default: True)

    Example:
        >>> myprint("Training epoch 1", doit=fabric.is_global_zero)
    """
    if doit:
        print(s, end=end, flush=flush)


def myprogbar(
    iterator: Any,
    desc: Optional[str] = None,
    doit: bool = True,
    ncols: int = 80,
    ascii: bool = True,
    leave: bool = True,
) -> tqdm:
    """
    Create progress bar with standard settings for distributed training.

    Args:
        iterator: Iterable to wrap with progress bar
        desc: Description to display (default: None)
        doit: Whether to show progress bar (default: True)
        ncols: Progress bar width in characters (default: 80)
        ascii: Use ASCII characters instead of Unicode (default: True)
        leave: Leave progress bar after completion (default: True)

    Returns:
        tqdm progress bar iterator

    Example:
        >>> for batch in myprogbar(dataloader, desc="Training", doit=is_rank_zero):
        ...     # Process batch
    """
    return tqdm(
        iterator,
        desc=desc,
        ascii=ascii,
        ncols=ncols,
        disable=not doit,
        leave=leave,
        file=sys.stdout,
        mininterval=0.2,
        maxinterval=2,
    )


def flush(doit: bool = True) -> None:
    """
    Conditionally flush stdout buffer.

    Args:
        doit: Whether to perform flush (default: True)

    Example:
        >>> flush(doit=fabric.is_global_zero)
    """
    if doit:
        sys.stdout.flush()


###################################################################################################


def report(
    dict: Dict[str, Any],
    desc: Optional[str] = None,
    ncols: int = 120,
    fmt: Optional[Dict[str, str]] = None,
    fmt_default: Dict[str, str] = {
        "loss": ".3f",
        "l_main": ".3f",
        "MAP": "5.3f",
        "m_MAP": "5.3f",
        "MR1": "7.1f",
        "m_MR1": "7.1f",
        "ARP": "5.2f",
        "m_ARP": "5.2f",
    },
    fmt_base: str = ".3f",
    clean_line: bool = True,
) -> str:
    """
    Format metrics dictionary into human-readable string with custom formatting.

    Creates nicely formatted string of key-value pairs with metric-specific
    number formatting. Supports cleaning previous line output for dynamic updates.

    Args:
        dict: Dictionary of metric names to values
        desc: Optional description prefix (default: None)
        ncols: Number of columns for line clearing (default: 120)
        fmt: Custom format strings for specific keys (default: None)
        fmt_default: Default format strings for known metrics
        fmt_base: Fallback format string for unknown metrics (default: ".3f")
        clean_line: Whether to clear previous line before printing (default: True)

    Returns:
        Formatted string ready for printing

    Example:
        >>> metrics = {'loss': 0.542, 'MAP': 0.8234, 'MR1': 15.3}
        >>> print(report(metrics, desc="Epoch 1"))
        Epoch 1:  MAP = 0.823,  MR1 =    15.3,  loss = 0.542
    """
    if clean_line:
        s = "\r" + " " * ncols + "\r"
    else:
        s = ""
    if desc is not None:
        s += desc + ":  "
    keys = list(dict.keys())
    keys.sort()
    for i, key in enumerate(keys):
        value = dict[key]
        if i > 0:
            s += ",  "
        s += key + " = "
        if type(value) == str:
            s += value
        else:
            if fmt is not None and key in fmt:
                ff = fmt[key]
            elif key in fmt_default:
                ff = fmt_default[key]
            else:
                ff = fmt_base
            aux = "{:" + ff + "}"
            s += aux.format(value)
    return s


###################################################################################################


class Timer:
    """
    Simple timer for tracking elapsed time in training loops.

    Provides formatted time strings (DD:HH:MM:SS or DD:HH:MM:SS.S) for
    monitoring training duration.

    Attributes:
        use_milliseconds: Whether to display milliseconds/decimals
        tstart: Start time (set by reset())

    Example:
        >>> timer = Timer()
        >>> # ... training loop ...
        >>> print(f"Training time: {timer.time()}")
        Training time: 00:15:42
    """

    def __init__(self, use_milliseconds: bool = False) -> None:
        """
        Initialize timer and start counting.

        Args:
            use_milliseconds: If True, shows decimals in seconds field (default: False)
        """
        self.use_milliseconds = use_milliseconds
        self.reset()

    def reset(self) -> None:
        """Reset timer to current time."""
        self.tstart = time.time()

    def time(self) -> str:
        """
        Get formatted elapsed time string.

        Returns:
            Formatted time as "HH:MM:SS" or "DD:HH:MM:SS.S" depending on
            elapsed time and use_milliseconds setting

        Example:
            >>> timer = Timer()
            >>> time.sleep(65)
            >>> timer.time()
            '00:01:05'
        """
        elapsed = time.time() - self.tstart
        msecs = elapsed % 60
        secs = int(elapsed) % 60
        mins = (int(elapsed) // 60) % 60
        hours = (int(elapsed) // (60 * 60)) % 24
        days = int(elapsed) // (60 * 60 * 24)
        if self.use_milliseconds:
            s = f"{msecs:04.1f}"
        else:
            s = f"{secs:02d}"
        s = f"{hours:02d}:{mins:02d}:" + s
        if days > 0:
            s = f"{days:02d}:" + s
        return s


###################################################################################################
