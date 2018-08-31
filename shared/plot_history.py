import matplotlib
import os
import platform

def has_display():
    if platform.system().lower() in ['darwin', 'windows', 'linux'] and "DISPLAY" in os.environ:
        return True
    else:
        return False

if not has_display():
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_metric(history, ax, met_idx="loss", met_name="Loss"):
    loss = history.history[met_idx]
    epochs = range(1, len(loss) + 1)
    ax.plot(epochs, loss, 'r-', label='Training')
    val_idx = 'val_{}'.format(met_idx)
    if val_idx in history.history:
        val_loss = history.history['val_{}'.format(met_idx)]
        ax.plot(epochs, val_loss, 'b-', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(met_name)

def plot_all(history_data, metrics = {'acc': 'Accuracy'}):
    if not isinstance(history_data, list):
        history_data = [history_data]
    metric_count = len(metrics) + 1
    col = 1
    combined = plt.figure()
    folds = len(history_data)

    for fold in range(folds):
        # plot loss
        row = 0
        plt_idx = row * folds + fold + 1
        ax1 = combined.add_subplot(metric_count, folds, plt_idx)
        print("Plotting Loss for fold {} with index of {}".format(fold, plt_idx))
        plot_metric(history_data[fold], ax1)

        # plot other metrics
        for idx, name in metrics.items():
            row += 1
            plt_idx = row * folds + fold + 1
            axn = combined.add_subplot(metric_count, folds, plt_idx)
            print("Plotting {} for fold {} with index of {}".format(name, fold, plt_idx))
            plot_metric(history_data[fold], axn, idx, name)
    plt.tight_layout(1.)
    plt.legend()
    if has_display():
        combined.show()
    return combined
