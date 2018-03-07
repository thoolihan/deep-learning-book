import matplotlib.pyplot as plt

def plot_metric(history, ax, met_idx="loss", met_name="Loss"):
    loss = history.history[met_idx]
    val_loss = history.history['val_{}'.format(met_idx)]
    epochs = range(1, len(loss) + 1)
    ax.plot(epochs, loss, 'r-', label='Training')
    ax.plot(epochs, val_loss, 'b-', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(met_name)
    ax.legend()

def plot_all(history):
    combined = plt.figure()
    ax1 = combined.add_subplot(2,1,1)
    plot_metric(history, ax1)
    ax2 = combined.add_subplot(2,1,2)
    plot_metric(history, ax2, 'acc', 'Accuracy')
    combined.show()
    return combined