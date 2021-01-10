from sklearn.decomposition import PCA
import sys

from graph_algoritms.generate_graph_tsne import generate_tsne_graph
from data.datasets import Mnist, FashionMnist, Cifar10, Cifar100, Fmow

# generate_pca_graph function generate and saves pca and t-sne method graph

def generate_pca_tsne_graph(dataset_name, data, labels):

    pca = PCA(n_components=50)
    embedding = pca.fit_transform(data)

    generate_tsne_graph(dataset_name=dataset_name, data=embedding, labels=labels, structure_name="pca_tsne")

    return

if __name__ == "__main__":
    print("Running dimensional-reduction-generator:")

    dataset_list = [Mnist(), FashionMnist(), Cifar10(), Cifar100(), Fmow()]

    dataset = dataset_list[4]
    if not (dataset is dataset_list[4]):
        dataset.load()
    data, labels = dataset.get()

    print("Running PCA_TSNE algorithm!")
    #for dataset_name in datasets_names:
        # dataset_name = datasets_names[i]

    generate_pca_tsne_graph(dataset_name=str(dataset), data=data, labels=labels)

    print("Done!")
    sys.exit()
