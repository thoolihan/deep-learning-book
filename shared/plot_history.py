import matplotlib.pyplot as plt

def plot_metric(history, fig=plt, met_idx="loss", met_name="Loss"):
    loss = history.history[met_idx]
    val_loss = history.history['val_{}'.format(met_idx)]
    epochs = range(1, len(loss) + 1)
    fig.plot(epochs, loss, 'r-o', label='Training')
    fig.plot(epochs, val_loss, 'b-o', label='Validation')
    fig.set_xlabel('Epochs')
    fig.set_ylabel(met_name)
    fig.legend()
    return fig

def plot_all(history):
    combined = plt.figure()
    ax1 = combined.add_subplot(2,1,1)
    plot_metric(history, ax1)
    ax2 = combined.add_subplot(2,1,2)
    plot_metric(history, ax2, 'acc', 'Accuracy')
    combined.show()
    return combined