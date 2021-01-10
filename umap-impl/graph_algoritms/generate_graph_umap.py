import umap
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

from data.datasets import Mnist, FashionMnist, Cifar10, Cifar100, Fmow
from params import path_graphs, graph_format

# generate_pca_graph function generate and saves umap method graph

def generate_umap_graph(dataset_name, data, labels, r_state=42, n_ngb=15, m_dst=0.1, metric="euclidean",
                        is_supervised=False, structure_name="umap"):

    if is_supervised:
        file_string = path_graphs + "graphs/" + structure_name + "/" + dataset_name + "/" + "supervised/" +\
                      structure_name + "_" + dataset_name + "_n_neighbors=" + str(n_ngb) + "_min_dist=" + str(m_dst)\
                      + "_metric=" + str(metric) + graph_format
    else:
        file_string = path_graphs + "graphs/" + structure_name + "/" + dataset_name + "/" + "unsupervised/" +\
                      structure_name + "_" + dataset_name + "_n_neighbors=" + str(n_ngb) + "_min_dist=" + str(m_dst)\
                      + "_metric=" + str(metric) + graph_format

    if os.path.isfile(file_string):
        print("File already exist: "+file_string)
        return

    reducer = umap.UMAP(random_state=r_state, n_neighbors=n_ngb, min_dist=m_dst, metric=metric)

    if is_supervised:
        embedding = reducer.fit_transform(data, y=labels)
    else:
        embedding = reducer.fit_transform(data)

    sns.set(context="paper", style="white")

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=0.1
    )
    plt.setp(ax, xticks=[], yticks=[])
    plt.axis("off")
    # plt.title(str(dataset_name)+" data embedded into two dimensions by UMAP", fontsize=18)

    fig.savefig(
        file_string,
        bbox_inches="tight")

    plt.close(fig)

    print("UMAP plot saved as " + file_string)

    return

if __name__ == "__main__":
    print("Running dimensional-reduction-generator:")

    dataset_list = [Mnist(), FashionMnist(), Cifar10(), Cifar100(), Fmow()]

    # for i in [0, 1, 2, 3, 4]:
    dataset = dataset_list[0]
    if not (dataset is dataset_list[4]):
        dataset.load()
    data, labels = dataset.get()

    # default states
    r_state = 42
    n_ngb = 15
    m_dst = 0.1
    metric = "euclidean"
    supervised = False

    print("Running UMAP algorithm!")

    """
    #  for metric in range(6,22):
    for n_ngb in [15]:  # range(5,55,5):
        for m_dst in [0.1]:  # [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
            #supervised = True
    """

    generate_umap_graph(dataset_name=str(dataset), data=data, labels=labels, n_ngb=15, is_supervised=True)

    print("Done!")
    sys.exit()
