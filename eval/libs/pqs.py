import math
import os
import re
from enum import IntEnum

import libs.common as common


class Algo(IntEnum):
    WARP_SELECT = 0
    BLOCK_SELECT = 1
    GRID_SELECT = 2


def _run_algo(algo: Algo, BATCH: int, N: int, K: int) -> float:
    cmd = f"{common.PQ_BIN} {algo.value} {BATCH} {N} {K}"
    elapsed: float = None

    with os.popen(cmd, 'r') as fout:
        pattern = re.compile(r"^elapsed: (\S+) ms$")
        for line in fout:
            line = line.strip()
            m = re.match(pattern, line)
            if m:
                elapsed = float(m.group(1))
                break

    return elapsed


def run_warp_select(BATCH: int, N: int, K: int) -> float:
    if K > 1024:
        return math.nan
    return _run_algo(Algo.WARP_SELECT, BATCH, N, K)

def run_block_select(BATCH: int, N: int, K: int) -> float:
    if K > 1024:
        return math.nan
    return _run_algo(Algo.BLOCK_SELECT, BATCH, N, K)

def run_grid_select(BATCH: int, N: int, K: int) -> float:
    if K > 512:
        return math.nan
    return _run_algo(Algo.GRID_SELECT, BATCH, N, K)


if __name__ == '__main__':
    t = run_warp_select(16, 18, 256)
    print(t)
    t = run_block_select(16, 18, 256)
    print(t)
    t = run_grid_select(16, 18, 256)
    print(t)
