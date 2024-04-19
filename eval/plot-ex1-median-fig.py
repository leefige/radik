#! /usr/bin/env python3
import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np

import libs.common as common

plt.rcParams.update({'font.size': 19})


START = 16
END = 23

data = []

with open(os.path.join(common.PLOT_DIR, "ex1-median.csv"), 'r', encoding='utf-8') as fin:
    reader = csv.reader(fin)
    # skip header
    next(reader, None)
    for line in reader:
        line = [float(s) for s in line]
        data.append(line)

# data = np.array(data)

x = [f"$2^{{{i:d}}}$" for i in range(START, END)]


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

plt.xlabel("input length $n$ of each task")
plt.ylabel("latency (ms)")
# plt.ylim(0, 17)
plt.plot(x, data[0], marker='s', label=r'$k=512$')
plt.plot(x, data[1], marker='D', linestyle='--', label=r'$k=\lfloor n/100 \rfloor$')
plt.plot(x, data[2], marker='v', linestyle=':', label='$k=n/4$')
plt.plot(x, data[3], marker='o', linestyle='-.', label='$k=n/2$')
plt.legend()
plt.tight_layout()
outname = os.path.join(common.PLOT_DIR, "ex1-median.png")
plt.savefig(outname)
print(f"Saved at {outname}")
