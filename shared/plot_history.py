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
    ax.legend()

def plot_all(history, metrics = {'acc': 'Accuracy'}):
    metric_count = 1+len(metrics)
    i = 1
    combined = plt.figure()
    # plot loss
    ax1 = combined.add_subplot(metric_count,1,i)
    plot_metric(history, ax1)

    # plot other metrics
    for idx, name in metrics.items():
        i += 1
        axn = combined.add_subplot(metric_count,1,i)
        plot_metric(history, axn, idx, name)
    plt.tight_layout(1.)
    combined.show()
    return combined