import os
import re
from enum import IntEnum

import libs.common as common


class AblationType(IntEnum):
    BASELINE = 0
    ONLY_1 = 1
    ONLY_2 = 2
    ONLY_3 = 3
    EXCEPT_1 = 4
    EXCEPT_2 = 5
    EXCEPT_3 = 6
    ALL = 7


def run_ablation(BATCH: int, N: int, K: int,
                ablation_type: AblationType) -> float:
    arg_ablation = ablation_type.value

    cmd = f"{common.ABLATION_BIN} {arg_ablation} {BATCH} {N} {K}"
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
    t = run_ablation(128, 23, 256, 0)
    print(t)
