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
    # zipf(N, 1.1), w/o scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.ZIPF_1_1, scaling=False))
    # zipf(N, 1.1), w/ scaling
    res.append(fn(Algo.RADIK)(BATCH, N, K, distribution=Distribution.ZIPF_1_1, scaling=True))
    # zipf(N, 1.1), pq-block
    res.append(fn(Algo.PQ_BLOCK)(BATCH, N, K, distribution=Distribution.ZIPF_1_1))
    # zipf(N, 1.1), pq-grid
    res.append(fn(Algo.PQ_GRID)(BATCH, N, K, distribution=Distribution.ZIPF_1_1))
    return res


def main():
    res = []
    for N in range(21, 30):
        print(f"Eval: N={N}, K={K}")
        res.append(test_case(N))
    res = np.array(res)
    res = res.T
    print("Done")

    filename = os.path.join(common.PLOT_DIR, "ex3-zipf-b.csv")
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
