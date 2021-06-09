"""
Plot a Given a training log file.
"""
import csv
import matplotlib.pyplot as plt


def plot_acc_log(training_log, nb_classes, savefig=None):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies_t = []
        accuracies_v = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            accuracies_t.append(float(acc))
            accuracies_v.append(float(val_acc))
            cnn_benchmark.append(1./nb_classes)  # random

        plt.figure(figsize=(5, 4))
        plt.plot(accuracies_t, label="Acc CNN train")
        plt.plot(accuracies_v, label="Acc CNN valid")
        plt.plot(cnn_benchmark, label="Acc random")
        plt.title(f"UCF-{nb_classes} accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()


def plot_loss_log(training_log, nb_classes, savefig=None):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        losses_t = []
        losses_v = []
        for epoch, acc, loss, val_acc, val_loss, in reader:
            losses_t.append(float(loss))
            losses_v.append(float(val_loss))

        plt.figure(figsize=(5, 4))
        plt.plot(losses_t, label="Loss CNN train")
        plt.plot(losses_v, label="Loss CNN valid")
        plt.title(f"UCF-{nb_classes} losses")
        plt.xlabel("Epoch")
        plt.ylabel("CCE Loss")
        plt.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()
