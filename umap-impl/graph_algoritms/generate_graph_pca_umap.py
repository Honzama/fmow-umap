from sklearn.decomposition import PCA
import sys

from graph_algoritms.generate_graph_umap import generate_umap_graph
from data.datasets import Mnist, FashionMnist, Cifar10, Cifar100, Fmow

# generate_pca_graph function generate and saves pca and umap method graph

def generate_pca_umap_graph(dataset_name, data, labels, is_supervised=False):

    pca = PCA(n_components=50)
    embedding = pca.fit_transform(data)

    generate_umap_graph(dataset_name=dataset_name, data=embedding, labels=labels, structure_name="pca_umap", is_supervised=is_supervised)

    return

if __name__ == "__main__":
    print("Running dimensional-reduction-generator:")

    dataset_list = [Mnist(), FashionMnist(), Cifar10(), Cifar100(), Fmow()]

    dataset = dataset_list[4]
    if not (dataset is dataset_list[4]):
        dataset.load()
    data, labels = dataset.get()

    print("Running PCA_UMAP algorithm!")
    #for dataset_name in datasets_names:
        # dataset_name = datasets_names[i]

    generate_pca_umap_graph(dataset_name=str(dataset), data=data, labels=labels, is_supervised=True)

    print("Done!")
    sys.exit()
