import sys
import time
from tqdm import tqdm

###################################################################################################


def myprint(s, end="\n", doit=True, flush=True):
    if doit:
        print(s, end=end, flush=flush)


def myprogbar(iterator, desc=None, doit=True, ncols=80, ascii=True, leave=True):
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


def flush(doit=True):
    if doit:
        sys.stdout.flush()


###################################################################################################


def report(
    dict,
    desc=None,
    ncols=120,
    fmt=None,
    fmt_default={
        "loss": ".3f",
        "l_main": ".3f",
        "MAP": "5.3f",
        "m_MAP": "5.3f",
        "MR1": "7.1f",
        "m_MR1": "7.1f",
        "ARP": "5.2f",
        "m_ARP": "5.2f",
    },
    fmt_base=".3f",
    clean_line=True,
):
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
    def __init__(self, use_milliseconds=False):
        self.use_milliseconds = use_milliseconds
        self.reset()

    def reset(self):
        self.tstart = time.time()

    def time(self):
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