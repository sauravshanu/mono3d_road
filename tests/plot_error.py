import collections
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot(seqs, gac_accuracy, accuracy):
    x = np.arange(len(seqs))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, accuracy, width, label='my')
    rects2 = ax.bar(x + width/2, gac_accuracy, width, label='gac')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('GAC vs My accuracies')
    ax.set_xticks(x)
    # ax.set_xticklabels(seqs)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    # plt.savefig('accuracies.png')


def load_json_accuracy(jsonfile):
    accuracies_file = json.load(open(jsonfile))
    gac_accuracy = accuracies_file['gac_accuracy']
    accuracy = accuracies_file['accuracy']
    seqs = accuracies_file['seqs']
    plot(seqs, gac_accuracy, accuracy)


def check_accuracies_v2(jsonfile):
    accuracies_file = json.load(open(jsonfile))
    gac_accuracy = accuracies_file['gac_accuracy']
    accuracy = accuracies_file['accuracy']
    seqs = accuracies_file['seqs']
    for acc, gac_acc, seq in zip(accuracy, gac_accuracy, seqs):
        if acc > gac_acc + 10:
            print('{}\n {} {}'.format(seq, acc, gac_acc))
    # plot(seqs, gac_accuracy, accuracy)


def check_accuracies(jsonfile):
    accuracies = json.load(open(jsonfile))
    for seq in accuracies:
        accuracy = accuracies[seq][0]
        gac_accuracy = accuracies[seq][1]
        if accuracy + 10 < gac_accuracy:
            print('Sequence {} has less than 10\% accuracy'.format(seq))
    # gac_accuracy = accuracies_file['gac_accuracy']
    # accuracy = accuracies_file['accuracy']
    # seqs = accuracies_file['seqs']
    # plot(seqs, gac_accuracy, accuracy)


def check_accuracies_custom():
    accuracies_file = json.load(open('accuracies_gac_my_without_gac.json'))
    gac_accuracy = accuracies_file['gac_accuracy']
    accuracy = accuracies_file['accuracy']
    seqs = accuracies_file['seqs']
    accuracies_file = json.load(open('images/accuracies.json'))
    gac_accuracy = accuracies_file['gac_accuracy']
    for acc, gac_acc, seq in zip(accuracy, gac_accuracy, seqs):
        if acc > gac_acc:
            print('{}\n {} {}'.format(seq, acc, gac_acc))
    # plot(seqs, gac_accuracy, accuracy)


if __name__ == '__main__':
    jsonfile = sys.argv[1]
    # load_json_accuracy(jsonfile)
    check_accuracies_v2(jsonfile)
    # check_accuracies_custom()
