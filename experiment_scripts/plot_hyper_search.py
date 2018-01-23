import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nfpexperiment.util as util
import numpy as np

def get_losses(loss_name):
    return map(lambda x : x[loss_name], jobs_data)

def get_hypers(hyper_name):
    return map(lambda x : x['varied_params'][hyper_name], jobs_data)

jobs_data = util.get_jobs_data(sys.argv[1:])

loss_types = ['train_loss', 'test_loss', 'halfway_train_loss']
color_list = ['RoyalBlue', 'DarkOliveGreen', 'DarkOrange',
              'MidnightBlue', 'DarkSlateGray', 'Red', 'Brown']
colors = dict(zip(loss_types, color_list))

hyper_names = jobs_data[0]['varied_params'].keys()
N_hypers = len(hyper_names)
fig = plt.figure(figsize=(3 * (N_hypers + 1), 3))

sorted_losses = sorted(filter(np.isfinite, get_losses('test_loss') + 
                                           get_losses('train_loss')))
ylims = [sorted_losses[0],
         sorted_losses[2 * len(sorted_losses)/3]]

for i, hyper_name in enumerate(hyper_names):
    ax = fig.add_subplot(1, (N_hypers + 1), i + 1)
    for loss_type in loss_types:
        ax.plot(get_hypers(hyper_name), 
                get_losses(loss_type), 'o', color=colors[loss_type])
        ax.set_title(hyper_name)
    ax.set_ylim(ylims)

# add_legend
lines = [mpl.lines.Line2D([0], [0], ls='', marker='o', color=colors[loss_type])
         for loss_type in loss_types]
fig.legend(lines, loss_types, 'lower right', frameon=False)

fig.savefig(sys.stdout, format='png')
