#! /usr/bin/env python3
import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np

import libs.common as common


plt.rcParams.update({'font.size': 19})

START = 4
END = 11

data = []

with open(os.path.join(common.PLOT_DIR, "2-batched-b.csv"), 'r', encoding='utf-8') as fin:
    reader = csv.reader(fin)
    # skip header
    next(reader, None)
    for line in reader:
        line = [float(s) for s in line]
        data.append(line)

data = np.array(data)

x = [f"${1 << i}$" for i in range(START, END)]
print(x)


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

# plt.title("same input length for all tasks in the batch")
plt.xlabel("$k$")
plt.ylabel("latency (ms)")
plt.ylim(0, 17)
plt.plot(x, data[0], marker='s', label='ours')
plt.plot(x, data[1], marker='D', linestyle='--', label='bitonic')
# plt.plot(x, data[2], marker='o', linestyle='-.', label='PQ-grid')
plt.plot(x, data[2], marker='v', linestyle=':', label='PQ-block')
plt.legend(loc=(0.545, 0.19))
plt.tight_layout()
outname = os.path.join(common.PLOT_DIR, "2-batched-b.png")
plt.savefig(outname)
print(f"Saved at {outname}")

