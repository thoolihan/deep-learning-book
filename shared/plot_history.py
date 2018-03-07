import matplotlib.pyplot as plt

def plot_loss(history, fig=plt):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    fig.plot(epochs, loss, 'r-o', label='Training')
    fig.plot(epochs, val_loss, 'b-o', label='Validation')
    fig.set_xlabel('Epochs')
    fig.set_ylabel('Loss')
    fig.legend()
    return fig

def plot_accuracy(history, fig=plt):
    loss = history.history['acc']
    val_loss = history.history['val_acc']
    epochs = range(1, len(loss) + 1)
    fig.plot(epochs, loss, 'r-o', label='Training')
    fig.plot(epochs, val_loss, 'b-o', label='Validation')
    fig.set_xlabel('Epochs')
    fig.set_ylabel('Accuracy')
    fig.legend()
    return fig

def plot_all(history):
    combined = plt.figure()
    ax1 = combined.add_subplot(2,1,1)
    plot_loss(history, ax1)
    ax2 = combined.add_subplot(2,1,2)
    plot_accuracy(history, ax2)
    combined.show()
    return combined