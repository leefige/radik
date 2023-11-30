#! /usr/bin/env python3
import csv
import math
import os.path
import re

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import libs.common as common

plt.rcParams.update({'font.size': 19})


npyname = os.path.join(common.PLOT_DIR, "1-simple-topk.npy")
data: np.ndarray = np.load(npyname, 'r')
data = data[1:]
rows, cols, algos = data.shape

Ns = [f"2^{{{n:d}}}" for n in range(22, 30)]
Ks = [f"{2 ** k}" for k in range(4, 13)]
Algos = ["RadiK", "bitonic", "PQ-grid", "PQ-block"]

assert rows == len(Ns)
assert cols == len(Ks)
assert algos == len(Algos)

# headers = ["Input length", "Method"] + Ks
# header_size = len(headers)

def draw_part(data, Ns, Ks, name):
    res:list[list[float]] = []
    for al in Algos:
        res.append([])

    # records = []
    for i in range(len(Ns)):
        row = data[i, ...]
        row: np.ndarray
        for record in row:
            record: np.ndarray
            record = record.tolist()
            assert len(record) == len(res)
            for k in range(len(Algos)):
                res[k].append(record[k])

            # record = [item for item in record]
            # record = [N, Algos[j]] + record
            # assert header_size == len(record)
            # records.append(record)

    x_labels = product(Ns, Ks)
    x_labels = [f"${a},{b}$" for a, b in x_labels]
    x = np.arange(len(x_labels))
    assert len(x) == len(res[0])

    # plt.figure(figsize=(3 * len(data), 6))

    plt.cla()
    fig, ax = plt.subplots()
    fig.set_figwidth(3 * len(data))
    fig.set_figheight(6)
    ax.yaxis.get_ticklocs(minor=True)
    # Initialize minor ticks
    ax.minorticks_on()
    # Now minor ticks exist and are turned on for both axes
    # Turn off x-axis minor ticks
    ax.xaxis.set_tick_params(which='minor', bottom=False)

    plt.grid(axis='y', which='minor', alpha=0.2)
    plt.grid(axis='y', which='major')
    plt.grid(axis='x', which='minor', visible=False)
    plt.grid(axis='x', which='major')
    # plt.title(label)
    plt.xlabel("test case ($n$, $k$)")
    plt.ylabel("latency (ms), logarithmic scale")

    n = len(Algos)
    width = 1
    factor = width * n + width

    plt.yscale('log')
    plt.bar(x * factor + width * 0, res[0], width=width, label='ours', color='white', edgecolor="C0", hatch='//')
    plt.bar(x * factor + width * 1, res[1], width=width, label=Algos[1], color='white', edgecolor="C1", hatch='--')
    plt.bar(x * factor + width * 2, res[2], width=width, label=Algos[2], color='white', edgecolor="C2", hatch='..')
    plt.bar(x * factor + width * 3, res[3], width=width, label=Algos[3], color='white', edgecolor="C3",)
    plt.xticks(x * factor + width * 1.5, x_labels, rotation=-60, fontsize=13.5)
    plt.xlim(-width * 2, len(x) * factor)

    plt.legend()
    plt.tight_layout()
    outname = os.path.join(common.PLOT_DIR, f"1-single-query-{name}.png")
    plt.savefig(outname)
    print(f"Saved at {outname}")

draw_part(data, Ns, Ks, "all")
