import math
import os
import re

import libs.common as common


def run_bitonic_select(BATCH: int, N: int, K: int) -> float:
    if K > 512:
        return math.nan

    # float
    data_type = 1
    # U[0, 1]
    distribution_type = 0

    cmd = f"{common.BITONIC_BIN} {data_type} {distribution_type} {K} {BATCH} {N} {N}"
    avg_time: float = None

    with os.popen(cmd, 'r') as fout:
        pattern = re.compile(r"^Bitonic TopK\s+averaged: (\S+) ms$")
        for line in fout:
            line = line.strip()
            m = re.match(pattern, line)
            if m:
                avg_time = float(m.group(1))
                break

    return avg_time * BATCH


if __name__ == '__main__':
    t = run_bitonic_select(16, 18, 256)
    print(t)
