#! /usr/bin/env python3
import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np

import libs.common as common


plt.rcParams.update({'font.size': 19})

data = []

with open(os.path.join(common.PLOT_DIR, "3-skewed.csv"), 'r', encoding='utf-8') as fin:
    reader = csv.reader(fin)
    next(reader, None)
    for line in reader:
        line = [float(s) for s in line]
        data.append(line)

data = np.array(data)

ylim = data.max() * 1.1

x = [f"$2^{{{i:d}}}$" for i in range(21, 30)]
print(x)


def draw(da, s, label):
    plt.cla()
    fig, ax = plt.subplots()
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
    plt.xlabel("input length")
    plt.ylabel("latency (ms)")
    plt.ylim(0, ylim)
    plt.plot(x, da[1], marker='s', label='with scaling')
    plt.plot(x, da[0], marker='v', linestyle='dashed', label='without scaling')
    plt.legend()
    plt.tight_layout()
    plt.savefig(s)


outname = os.path.join(common.PLOT_DIR, "3-skewed-a.png")
draw(data[:2], outname, "inputs sampled from $\\mathrm{{Uniform}}[0.6,0.7]$")
print(f"Saved at {outname}")

outname = os.path.join(common.PLOT_DIR, "3-skewed-b.png")
draw(data[2:], outname,
     "inputs sampled from $\\mathrm{{Uniform}}[128.6,128.7]$")
print(f"Saved at {outname}")
