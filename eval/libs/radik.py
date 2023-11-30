import os
import re
from enum import IntEnum

import libs.common as common


class Distribution(IntEnum):
    U_0_1 = 0
    U_0_6_0_7 = 1
    U_128_6_128_7 = 2


def run_radik(BATCH: int, N: int, K: int,
              scaling: bool = False,
              rand_task_len: bool = False,
              distribution: Distribution = Distribution.U_0_1,
              batched: bool = True,
              padding: bool = True) -> float:
    arg_scaling = 1 if scaling else 0
    arg_rand_task_len = 1 if rand_task_len else 0
    arg_distribution = distribution.value
    arg_batched = 1 if batched else 0
    arg_padding = 1 if padding else 0

    cmd = f"{common.RADIK_BIN} {BATCH} {N} {K} {arg_rand_task_len} "\
          f"{arg_scaling} {arg_distribution} {arg_batched} {arg_padding}"
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
    t = run_radik(16, 18, 256)
    print(t)
