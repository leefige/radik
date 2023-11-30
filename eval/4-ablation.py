#! /usr/bin/env python3
import csv
import os.path
from typing import List

import numpy as np

import libs.common as common
from libs.ablation import AblationType, run_ablation

BATCH = 16
N = 20
ITER = 5


def test_case(K: int) -> List[float]:
    res = []
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.BASELINE))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.ONLY_1))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.ONLY_2))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.ONLY_3))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.EXCEPT_1))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.EXCEPT_2))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.EXCEPT_3))
    res.append(run_ablation(BATCH, N, K, ablation_type=AblationType.ALL))
    return res


def main():
    K = 2048
    res = []
    for _ in range(ITER):
        print(f"Eval: BATCH={BATCH}, N={N}, K={K}")
        res.append(test_case(K))
    res = np.array(res)
    res = res.T
    res = res.mean(axis=1)
    print("Done")

    filename = os.path.join(common.PLOT_DIR, "4-ablation.csv")
    with open(filename, 'w', encoding='utf-8') as fout:
        fieldnames = [f"{AblationType(i).name}" for i in range(8)]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        # for row in res:
        content = dict(zip(fieldnames, res))
        writer.writerow(content)
    print(f"Saved at {filename}")


if __name__ == '__main__':
    main()
