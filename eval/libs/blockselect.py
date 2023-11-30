import math
import os
import re

import libs.common as common


def run_block_select(BATCH: int, N: int, K: int) -> float:
    if K > 2048:
        return math.nan

    cmd = f"{common.BLOCKSELECT_BIN} {BATCH} {N} {K}"
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


if __name__ == '__main__':
    t = run_block_select(16, 18, 256)
    print(t)
