import matplotlib.pyplot as plt

def plot_training_history(history, figsize=(12, 5), lang='de'):
    """
    Plots training and validation accuracy and loss over epochs.

    Args:
        history (keras.callbacks.History): Training history returned by model.fit().
        figsize (tuple): Size of the overall figure.
        lang (str): Language for labels ('de' for German, 'en' for English).
    """
    if lang == 'de':
        acc_label = "Genauigkeit"
        val_acc_label = "Validierungsgenauigkeit"
        loss_label = "Verlust"
        val_loss_label = "Validierungsverlust"
        epoch_label = "Epoche"
        acc_title = "Modellgenauigkeit über Epochen"
        loss_title = "Modellverlust über Epochen"
    else:
        acc_label = "Training Accuracy"
        val_acc_label = "Validation Accuracy"
        loss_label = "Training Loss"
        val_loss_label = "Validation Loss"
        epoch_label = "Epoch"
        acc_title = "Model Accuracy Over Epochs"
        loss_title = "Model Loss Over Epochs"

    plt.figure(figsize=figsize)

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label=acc_label)
    plt.plot(history.history["val_accuracy"], label=val_acc_label)
    plt.xlabel(epoch_label)
    plt.ylabel("Accuracy")
    plt.title(acc_title)
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label=loss_label)
    plt.plot(history.history["val_loss"], label=val_loss_label)
    plt.xlabel(epoch_label)
    plt.ylabel("Loss")
    plt.title(loss_title)
    plt.legend()

    plt.tight_layout()
    plt.show()