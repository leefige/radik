#! /usr/bin/env python3
import csv
import os.path
from typing import List

import numpy as np

import libs.common as common
from libs.algos import ALGOS, Algo
from libs.common import Distribution

BATCH = 1
K = 512


def fn(algo: Algo):
    return ALGOS[algo][1]


def test_case(N: int) -> List[float]:
    res = []
    # U[0.6, 0.7], w/o scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.U_0_6_0_7, scaling=False))
    # U[128.6, 128.7], w/o scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.U_128_6_128_7, scaling=False))
    # U[0.6, 0.7], w/ scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.U_0_6_0_7, scaling=True))
    # U[128.6, 128.7], w/ scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.U_128_6_128_7, scaling=True))
    # U[0.6, 0.7], pq-block
    res.append(fn(Algo.PQ_BLOCK)(BATCH, N, K, distribution=Distribution.U_0_6_0_7))
    # U[128.6, 128.7], pq-block
    res.append(fn(Algo.PQ_BLOCK)(BATCH, N, K, distribution=Distribution.U_128_6_128_7))
    # U[0.6, 0.7], pq-grid
    res.append(fn(Algo.PQ_GRID)(BATCH, N, K, distribution=Distribution.U_0_6_0_7))
    # U[128.6, 128.7], pq-grid
    res.append(fn(Algo.PQ_GRID)(BATCH, N, K, distribution=Distribution.U_128_6_128_7))
    return res


def main():
    res = []
    for N in range(21, 30):
        print(f"Eval: N={N}, K={K}")
        res.append(test_case(N))
    res = np.array(res)
    res = res.T
    print("Done")

    filename = os.path.join(common.PLOT_DIR, "ex2-skewed.csv")
    with open(filename, 'w', encoding='utf-8') as fout:
        fieldnames = [f"2^{i}" for i in range(21, 30)]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in res:
            content = dict(zip(fieldnames, row))
            writer.writerow(content)
    print(f"Saved at {filename}")


if __name__ == '__main__':
    main()
