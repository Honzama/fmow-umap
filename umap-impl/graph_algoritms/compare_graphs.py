import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from params import path_graphs, str_mnist, str_fashion_mnist, str_cifar10, str_cifar100, str_fmow, graph_format

# generate_graph_comparison function generates and show graph collage

def generate_graph_comparison(data, fig_size, fig_shape, x_titles, y_titles, file_name="test", font_size=20, save_fig=False):
    fig = plt.figure(figsize=fig_size, dpi=300)

    n_rows = fig_shape[0]
    n_col = fig_shape[1]
    x_pos = 0
    y_pos = 0
    for i in range(1, n_col * n_rows + 1):
        if x_pos >= n_col:
            x_pos = 0
            y_pos += 1

        plt.subplot(n_rows, n_col, i)
        image = mpimg.imread(data[i-1])
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        if y_pos == 0:
            plt.title(x_titles[x_pos], fontsize=font_size)
        if x_pos == 0:
            plt.ylabel(y_titles[y_pos], fontsize=font_size)

        x_pos += 1

    if save_fig:
        plt.savefig(path_graphs + "graphs/" + file_name + ".png", dpi=fig.dpi,
                    bbox_inches="tight")

    plt.show()
    plt.close()

# generate_graph_path function generates file path of already generated graphs in graphs file structure

def generate_graph_path(graph_method, dataset, is_supervised=False, fmow_size=10, n_ngb=15, m_dst=0.1, metric="euclidean"):
    path_str = path_graphs + "graphs/" + graph_method + "/" + dataset + "/"

    if graph_method == "umap" or graph_method == "pca_umap":
        if is_supervised:
            path_str += "supervised/"
        else:
            path_str += "unsupervised/"

    if dataset == str_fmow:
        path_str += str(fmow_size) + "x" + str(fmow_size) + "/"

    path_str += graph_method + "_" + dataset

    if graph_method == "umap" or graph_method == "pca_umap":
        path_str += "_n_neighbors=" + str(n_ngb) + "_min_dist=" + str(m_dst)\
                      + "_metric=" + str(metric)

    path_str += graph_format

    return path_str


if __name__ == "__main__":
    graph_methods = ["pca", "tsne", "pca_tsne", "umap", "pca_umap"]
    datasets = [str_mnist, str_fashion_mnist, str_cifar10, str_cifar100, str_fmow]

    graphs = []

    x_titles = [
        "UMAP",
        "t-SNE",
        "UMAP"
    ]

    y_titles = []
    size = 10

    # 0 - datasets graph
    # 1 - fmow graph
    graph_mod = 1
    graph_index = graph_methods.index("pca")
    second_line = False
    third_line = False
    fmow_sizes = [10, 50, 100]  # [10, 50, 75]

    # datasets graph
    if graph_mod == 0:

        for dataset in datasets:
            if dataset is not datasets[4]:
                y_titles.append(dataset)
            else:
                y_titles.append(datasets[4] + " (" + str(size) + "x" + str(size) + ")")

            graphs.append(generate_graph_path(graph_methods[graph_index], dataset, is_supervised=False, fmow_size=size))

        if second_line:
            for dataset in datasets:
                if dataset is not datasets[4]:
                    y_titles.append(dataset)
                else:
                    y_titles.append(datasets[4] + " (" + str(size) + "x" + str(size) + ")")

                graphs.append(generate_graph_path(graph_methods[graph_index+1], dataset, is_supervised=False, fmow_size=size))

        if third_line:
            for dataset in datasets:
                if dataset is not datasets[4]:
                    y_titles.append(dataset)
                else:
                    y_titles.append(datasets[4] + " (" + str(size) + "x" + str(size) + ")")

                graphs.append(generate_graph_path(graph_methods[graph_index+3], dataset, is_supervised=False, fmow_size=size))

    # fmow graph
    if graph_mod == 1:

        for size in fmow_sizes:
            graphs.append(generate_graph_path(graph_methods[graph_index], datasets[4], is_supervised=False, fmow_size=size))
            # graphs.append(generate_graph_path(graph_methods[3], datasets[4], is_supervised=True, fmow_size=size))

            y_titles.append(datasets[4] + " (" + str(size) + "x" + str(size) + ")")

    # y_titles[4] += " (10x10)"
    #
    # (10, 5) 1x2
    # (14, 5)   1x3
    # (22, 5)   1x5
    # (22, 8)   2x5
    # (18, 9)  3x5
    #
    # (6.5, 10) 2x1
    # (13, 11)  2x2
    # (6, 14)   3x1
    # (12, 20)  4x2
    # (6, 22)   5x1
    # (11, 22)  5x2
    # (13, 18)  5x3
    #
    # (42, 50)
    # (14, 10)

    generate_graph_comparison(graphs, (14, 5), (1, 3), y_titles, x_titles, save_fig=False)

    print("Done!")
    sys.exit()
