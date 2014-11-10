import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_predictions(results_filename, outdir):

    def scatterplot(x, y, title, outfilename):
        fig = plt.figure()
        fig.add_subplot(1,1,1)
        plt.scatter(x, y)
        matplotlib.rcParams.update({'font.size': 16})
        plt.xlabel("predicted " + target_name)
        plt.ylabel("true " + target_name)
        plt.title(title)
        plt.savefig(os.path.join(outdir, outfilename + '.png'))
        plt.savefig(os.path.join(outdir, outfilename + '.eps'))
        plt.draw()
        plt.close()

    preds = np.load(results_filename)
    target_name = str(preds['target_name'])
    scatterplot(preds['train_preds'], preds['train_targets'],
                "Training Accuracy on " + target_name, "training_" + target_name)
    scatterplot(preds['test_preds'], preds['test_targets'],
                "Test Accuracy on " + target_name, "testing_" + target_name)
