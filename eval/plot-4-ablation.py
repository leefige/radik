#! /usr/bin/env python3
import csv
import os.path

import matplotlib.pyplot as plt
import numpy as np

import libs.common as common


plt.rcParams.update({'font.size': 19})

data = []

with open(os.path.join(common.PLOT_DIR, "4-ablation.csv"), 'r', encoding='utf-8') as fin:
    reader = csv.reader(fin)
    # skip header
    next(reader, None)
    for line in reader:
        line = [float(s) for s in line]
        data.append(line)

data = np.array(data)

x_labels = [
    'baseline',
    'only\n(1)',
    'only\n(2)',
    'only\n(3)',
    'except\n(1)',
    'except\n(2)',
    'except\n(3)',
    'all',
]
x = np.arange(len(x_labels))


plt.cla()
# plt.figure(figsize=(9.6, 4.8))

fig, ax = plt.subplots()
fig.set_figwidth(9.6)
fig.set_figheight(4.8)


ax.yaxis.get_ticklocs(minor=True)
# Initialize minor ticks
ax.minorticks_on()
# Now minor ticks exist and are turned on for both axes
# Turn off x-axis minor ticks
ax.xaxis.set_tick_params(which='minor', bottom=False)

plt.grid(axis='y', which='minor', alpha=0.2)
plt.grid(axis='y', which='major')
plt.grid(axis='x', which='both', visible=False)
# plt.grid(axis='x', which='major')
# plt.title(label)
# plt.xlabel("$k$")
plt.ylabel("latency (ms)")

width = 1

plt.bar(x, data[0], width=0.8 * width, color='white', edgecolor="C0", hatch='//////')
# plt.bar(x * factor + width * 1, res[1], width=width, label='bitonic', color='white', edgecolor="C1", hatch='xxxxx')
plt.xticks(x, x_labels)
# plt.xlim(-width * 2, len(x))


# plt.legend()
plt.tight_layout()
outname = os.path.join(common.PLOT_DIR, "4-ablation.png")
plt.savefig(outname)
print(f"Saved at {outname}")
