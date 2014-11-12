import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

font = "sans-serif"
fig = plt.figure(figsize=(5,5))
ax = plt.axes([0,0,1,1])
ax.set_axis_off()

# Parameters
unit_radius = 0.025
unit_separation = unit_radius * 2.5
layer_height = 0.2
num_hidden_units = [5, 10, 4, 1]
layer_1_height = 0.2

def draw_unit(x, y, **kwargs):
    unit = mpatches.Circle((x, y), unit_radius, color="blue", zorder=3, **kwargs)
    ax.add_patch(unit)
    return unit

def draw_layer(x, y, N):
    unit_list = []
    for i in range(N):
        pos_x = x + (i - float(N - 1)/2) * unit_separation
        unit_list.append(draw_unit(pos_x, y))
    return unit_list

def draw_connecting_edges(unit_list_A, unit_list_B):
    for unit_A in unit_list_A:
        for unit_B in unit_list_B:
            x = [unit_A.center[0], unit_B.center[0]]
            y = [unit_A.center[1], unit_B.center[1]]
            line = mlines.Line2D(x, y, lw=1.5, color="green", zorder=1)
            ax.add_line(line)

prev_layer = None
for i, N in enumerate(num_hidden_units):
    cur_layer = draw_layer(0.5, layer_1_height + i * layer_height, N)
    if prev_layer:
        draw_connecting_edges(cur_layer, prev_layer)

    prev_layer = cur_layer

plt.savefig("nn_example.png")
plt.savefig("nn_example.pdf")
