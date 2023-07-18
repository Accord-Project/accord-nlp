# Created by Hansi at 29/06/2023

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_seq_length_histo(texts, plot_path=None):
    seq_lengths = [len(x.split(' ')) for x in texts]
    # print(seq_lengths)

    print(f'Min: {min(seq_lengths)}')
    print(f'Max: {max(seq_lengths)}')

    binwidth = 10
    density, bins, bars = plt.hist(seq_lengths, bins=range(0, max(seq_lengths) + binwidth, binwidth), rwidth=0.5,
                                   color='#607c8e')
    counts, _ = np.histogram(seq_lengths, bins)
    print(counts)
    print(bins)

    plt.xlabel('Sequence Length')
    plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(np.arange(0, max(seq_lengths) + binwidth, binwidth * 4))
    if plot_path is not None:
        plt.savefig(plot_path)
    plt.show()


if __name__ == '__main__':
    data_path = '../../data/ner/processed-entities.csv'
    plot_path = '../../data/ner/seq_histogram.png'

    df = pd.read_csv(data_path, encoding='utf-8')
    texts = df['processed_content'].tolist()
    plot_seq_length_histo(texts, plot_path=None)