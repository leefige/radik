#! /usr/bin/env python3

import os.path
from typing import List

import numpy as np

import libs.common as common
from libs.algos import ALGOS


def test_case(N: int, K: int) -> List[float]:
    res = []
    for _, fn in ALGOS.values():
        res.append(fn(1, N, K))
    return res

def main():
    res = []
    for N in range(21, 30):
        row = []
        for K_order in range(4, 13):
            K = 1 << K_order
            print(f"Eval: N={N}, K={K}")
            row.append(test_case(N, K))
        res.append(row)
    res = np.array(res)
    print("Done")

    filename = os.path.join(common.PLOT_DIR, "1-simple-topk.npy")
    np.save(filename, res)
    print(f"Saved at {filename}")


if __name__ == '__main__':
    main()
