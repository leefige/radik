#! /usr/bin/env python3
import csv
import os.path
from typing import List

import numpy as np

import libs.common as common
from libs.algos import ALGOS, Algo

START = 16
END = 23

BATCH = 16
K = 512


def fn(algo: Algo):
    return ALGOS[algo][1]


def test_case(N: int) -> List[float]:
    res = []
    res.append(fn(Algo.RADIK)(BATCH, N, K))
    res.append(fn(Algo.BITONIC)(BATCH, N, K))
    # res.append(fn(Algo.PQ_GRID)(BATCH, N, K))
    res.append(fn(Algo.PQ_BLOCK)(BATCH, N, K))
    return res


def main():
    res = []
    for N in range(START, END):
        print(f"Eval: BATCH={BATCH}, N={N}, K={K}")
        res.append(test_case(N))
    res = np.array(res)
    res = res.T
    print("Done")

    filename = os.path.join(common.PLOT_DIR, "2-batched-a.csv")
    with open(filename, 'w', encoding='utf-8') as fout:
        fieldnames = [f"2^{i}" for i in range(START, END)]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in res:
            content = dict(zip(fieldnames, row))
            writer.writerow(content)
    print(f"Saved at {filename}")


if __name__ == '__main__':
    main()
