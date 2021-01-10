from openTSNE import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

from data.datasets import Mnist, FashionMnist, Cifar10, Cifar100, Fmow
from params import path_graphs, graph_format

# generate_pca_graph function generate and saves t-sne method graph

def generate_tsne_graph(dataset_name, data, labels, structure_name="tsne"):

    file_string = path_graphs + "graphs/" + structure_name + "/" + dataset_name + "/" + structure_name + "_" + dataset_name + graph_format

    if os.path.isfile(file_string):
        print("File already exist: " + file_string)
        return

    tsne = TSNE()
    embedding = tsne.fit(data)

    sns.set(context="paper", style="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=0.1
    )
    plt.setp(ax, xticks=[], yticks=[])
    plt.axis("off")
    # plt.title(str(dataset_name)+" data embedded into two dimensions by TSNE", fontsize=18)

    fig.savefig(
        file_string,
        bbox_inches="tight")

    plt.close(fig)

    print("TSNE plot saved as " + file_string)

    return

if __name__ == "__main__":
    print("Running dimensional-reduction-generator:")

    dataset_list = [Mnist(), FashionMnist(), Cifar10(), Cifar100(), Fmow()]

    dataset = dataset_list[4]
    if not (dataset is dataset_list[4]):
        dataset.load()
    data, labels = dataset.get()

    print("Running TSNE algorithm!")
    #for dataset_name in datasets_names:
        # dataset_name = datasets_names[i]

    generate_tsne_graph(dataset_name=str(dataset), data=data, labels=labels)

    print("Done!")
    sys.exit()
