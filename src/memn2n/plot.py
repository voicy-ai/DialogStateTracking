from matplotlib import pyplot as plt
plt.style.use('ggplot')

import csv

LOG_FOLDER='log/'
PLOT_FOLDER='plots/'


def read_from_log(log_dir=LOG_FOLDER):
    epochs, tacc, vacc, tloss = [], [], [], []
    for i in range(1,7):
        f = open(log_dir + 'log.task{}.txt'.format(i))
        data = list(csv.reader(f, delimiter=' '))
        get_col = lambda idx : [i[idx] for i in data]
        epochs.append(get_col(0))
        tacc.append(get_col(1))
        vacc.append(get_col(2))
        tloss.append(get_col(3))
    return epochs, tacc, vacc, tloss


def plot(epochs, data, title):
    plt.clf()
    plt.title(title)
    for i in range(6):
        plt.plot(epochs[i], data[i], 
                 label='task#{}'.format(i+1),
                 linewidth=2)
    plt.legend()
    plt.savefig(PLOT_FOLDER + title + '.png', dpi=300)


if __name__ == '__main__':
    # read from log files
    epochs, tacc, vacc, tloss = read_from_log()
    # save plot to disk
    plot(epochs, tacc, 'Training Accuracy')
    plot(epochs, vacc, 'Validation Accuracy')
    plot(epochs, tloss, 'Training Loss')
