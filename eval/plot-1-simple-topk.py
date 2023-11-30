#! /usr/bin/env python3
import math
import os.path

import numpy as np
from tabulate import tabulate

import libs.common as common


npyname = os.path.join(common.PLOT_DIR, "1-simple-topk.npy")
data: np.ndarray = np.load(npyname, 'r')
rows, cols, algos = data.shape

Ns = [f"2^{n}" for n in range(21, 30)]
Ks = [f"k={2 ** k}" for k in range(4, 13)]
Algos = ["radik", "bitonic", "PQ-grid", "PQ-block"]

assert rows == len(Ns)
assert cols == len(Ks)
assert algos == len(Algos)

headers = ["Input length", "Method"] + Ks
header_size = len(headers)

records = []
for i, row in enumerate(data):
    row: np.ndarray
    N = Ns[i]
    tile = row.T
    for j, record in enumerate(tile):
        record: np.ndarray
        record = record.tolist()
        record = [item for item in record]
        record = [N, Algos[j]] + record
        assert header_size == len(record)
        records.append(record)

table = tabulate(records, headers=headers, tablefmt="outline",
                 numalign="right", floatfmt=".2f")
outname = os.path.join(common.PLOT_DIR, "1-simple-topk.txt")
with open(outname, 'w', encoding='utf-8') as fout:
    fout.write(table)

print(f"Saved at {outname}")
