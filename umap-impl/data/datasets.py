from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
import pickle
import numpy as np
from PIL import Image
import os
import sys
import h5py
import time
import json
import threading as th

from params import path_cifar10, path_cifar100, path_fmow_rgb, path_catch_files, fmow_category_names, fmow_image_size, \
    str_mnist, str_fashion_mnist, str_cifar10, str_cifar100, str_fmow

Image.MAX_IMAGE_PIXELS = None

# Mnist class envelope mnist dataset

class Mnist:

    def __init__(self):
        self.data = None
        self.labels = None
        self.str_name = str_mnist

    def __str__(self):
        return self.str_name

    # load function saves loaded data and labels into Mnist object

    def load(self):
        # mnist = fetch_mldata("MNIST original")
        mnist = fetch_openml("mnist_784", version=1)
        print("MNIST dataset loaded!")
        self.data = mnist.data.astype("uint8")
        self.labels = mnist.target.astype("int32")
        # np.array(list(map(int, fashion_mnist.target)))

    # get function returns data and labels saved in Mnist object

    def get(self):
        return self.data, self.labels

# FashionMnist class envelope fashion-mnist dataset

class FashionMnist:

    def __init__(self):
        self.data = None
        self.labels = None
        self.str_name = str_fashion_mnist

    def __str__(self):
        return self.str_name

    # load function saves loaded data and labels into FashionMnist object

    def load(self):
        # fashion_mnist = fetch_mldata("Fashion-MNIST")
        fashion_mnist = fetch_openml("Fashion-MNIST")
        print("Fashion-MNIST dataset loaded!")
        self.data = fashion_mnist.data.astype("uint8")
        self.labels = np.array(list(map(int, fashion_mnist.target)))

    # get function returns data and labels saved in FashionMnist object

    def get(self):
        return self.data, self.labels

# Cifar10 class envelope cifar-10 dataset

class Cifar10:

    def __init__(self):
        self.data = None
        self.labels = None
        self.path = path_cifar10
        self.str_name = str_cifar10

    def __str__(self):
        return self.str_name

    # __load_file function load file from folder_path

    def __load_file(self, folder_path):
        with open(folder_path, mode="rb") as file:
            batch = pickle.load(file, encoding="latin1")

        file_data = batch["data"]
        file_labels = batch["labels"]

        return file_data, file_labels

    # load function saves loaded data and labels into Cifar10 object

    def load(self):
        data, labels = self.__load_file(self.path + "/data_batch_" + str(1))
        print("CIFAR-10 batch number 1/5 loaded!")

        for i in range(2, 6):
            temp_data, temp_labels = self.__load_file(self.path + "/data_batch_" + str(i))
            data = np.concatenate((data, temp_data), axis=0)
            labels = np.concatenate((labels, temp_labels), axis=0)
            print("CIFAR-10 batch number " + str(i) + "/5 loaded!")

        test_data, test_labels = self.__load_file(self.path + "/test_batch")
        data = np.concatenate((data, test_data), axis=0)
        labels = np.concatenate((labels, test_labels), axis=0)
        print("CIFAR-10 test batch loaded!")

        print("CIFAR-10 dataset loaded!")

        self.data = data
        self.labels = labels

    # get function returns data and labels saved in Cifar100 object

    def get(self):
        return self.data, self.labels

# Cifar100 class envelope cifar-100 dataset

class Cifar100:

    def __init__(self):
        self.data = None
        self.labels = None
        self.path = path_cifar100
        self.str_name = str_cifar100

    def __str__(self):
        return self.str_name

    # __load_file function load file from folder_path

    def __load_file(self, folder_path):
        with open(folder_path, mode="rb") as file:
            batch = pickle.load(file, encoding="latin1")

        file_data = batch["data"]
        file_labels = batch["fine_labels"]

        return file_data, file_labels

    # load function saves loaded data and labels into Cifar100 object

    def load(self):
        data, labels = self.__load_file(self.path + "/train")
        print("CIFAR-100 train batch loaded!")

        test_data, test_labels = self.__load_file(self.path + "/test")
        data = np.concatenate((data, test_data), axis=0)
        labels = np.concatenate((labels, test_labels), axis=0)
        print("CIFAR-100 test batch loaded!")

        print("CIFAR-100 dataset loaded!")

        self.data = data
        self.labels = labels

    # get function returns data and labels saved in Cifar100 object

    def get(self):
        return self.data, self.labels

# Fmow class envelope fmow dataset

class Fmow:

    def __init__(self, str_files_group="train_rgb"):
        self.data = None
        self.labels = None
        self.path = path_fmow_rgb
        self.path_catch_files = path_catch_files
        self.category_names = fmow_category_names
        self.image_size = fmow_image_size
        self.file_name = "fmow-img_size=" + str(self.image_size)
        self.thread_couple = []
        self.str_name = str_fmow
        self.files_group = str_files_group

    def __str__(self):
        return self.str_name + ""

    # __scan_files function scan fmow dataset files structure

    def __scan_files(self, root_file):
        print("Scanning fmow dataset files structure")
        category_index = 0
        rgb_files = []
        msrgb_files = []

        while category_index < len(self.category_names):
            cat = self.category_names[category_index]
            path = self.path + "/" + root_file + "/" + cat

            dir_index = 0
            dir_found = 0
            number_of_dirs = len(next(os.walk(path))[1])
            while dir_found < number_of_dirs:

                if os.path.isdir(path + "/" + cat + "_" + str(dir_index)):
                    # print(str(category_index) + ": " + path + "/" + cat + "_" + str(dir_index))

                    dir_found += 1
                    file_index = 0
                    file_found = 0
                    number_of_files = len(next(os.walk(path + "/" + cat + "_" + str(dir_index)))[2])

                    for l in os.listdir(path + "/" + cat + "_" + str(dir_index)):
                        if "jpg." in l or "json." in l or ".jpg_tmp" in l or ".json_tmp" in l:
                            file_found += 1

                    while file_found < number_of_files:
                        meta_path = path + "/" + cat + "_" + str(dir_index) + "/" + cat + "" + "_" + str(
                            dir_index) + "_" + str(
                            file_index)

                        if os.path.isfile(meta_path + "_rgb" + ".jpg"):
                            rgb_files.append(meta_path + "_rgb")
                            file_found += 2

                        if os.path.isfile(meta_path + "_msrgb" + ".jpg"):
                            msrgb_files.append(meta_path + "_msrgb")
                            file_found += 2

                        file_index += 1

                dir_index += 1
            category_index += 1

        return rgb_files, msrgb_files

    # __load_file_list function open fmow dataset files structure file if it exist, or creates it

    def __load_file_list(self):
        if os.path.exists(self.path_catch_files + "fmow_file_list.hdf5"):
            print("Load file fmow_file_list.hdf5 exist!")
            hdf5_store = h5py.File(self.path_catch_files + "fmow_file_list.hdf5", "r")
            files = hdf5_store[self.files_group][:].astype(str)
            hdf5_store.close()
            return files
        else:
            print("Creating file fmow_file_list.hdf5")
            val_rgb_files, val_msrgb_files = self.__scan_files("val")
            train_rgb_files, train_msrgb_files = self.__scan_files("train")

            hdf5_store = h5py.File(self.path_catch_files + "fmow_file_list.hdf5", "a")
            hdf5_store.create_dataset("val_rgb", data=np.array(val_rgb_files, dtype="S"))
            hdf5_store.create_dataset("val_msrgb", data=np.array(val_msrgb_files, dtype="S"))
            hdf5_store.create_dataset("train_rgb", data=np.array(train_rgb_files, dtype="S"))
            hdf5_store.create_dataset("train_msrgb", data=np.array(train_msrgb_files, dtype="S"))
            hdf5_store.close()

            if self.files_group == "train_rgb":
                return train_rgb_files
            elif self.files_group == "train_msrgb":
                return train_msrgb_files
            elif self.files_group == "val_rgb":
                return val_rgb_files
            elif self.files_group == "val_msrgb":
                return val_msrgb_files

            return

    # __transform_thread function load and returns img and label from file_path

    def __transform_thread(self, file_path):
        img = Image.open(
            file_path + ".jpg")

        im_resized = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)

        img_pixels = self.image_size * self.image_size
        rgb_arr = np.reshape(np.reshape(np.array(im_resized), (img_pixels, 3)), (img_pixels * 3), order="F")

        with open(file_path + ".json") as json_file:
            json_temp = json.load(json_file)

        self.thread_couple.append([rgb_arr, self.category_names.index(json_temp["bounding_boxes"][0]["category"])])

    # load function creates img data and labels memory maps
    # and fill them with data from file list through paralel threads using __transform_thread function

    def load(self):
        files = self.__load_file_list()
        num_of_files = len(files)

        if not os.path.exists(self.path_catch_files + self.file_name + "-data(" + self.files_group + ").mymemmap"):
            data = np.memmap(self.path_catch_files + self.file_name + "-data(" + self.files_group + ").mymemmap",
                             dtype="uint8", mode="w+",
                             shape=(num_of_files, self.image_size * self.image_size * 3))
            labels = np.memmap(self.path_catch_files + self.file_name + "-labels(" + self.files_group + ").mymemmap",
                               dtype="int32", mode="w+",
                               shape=(num_of_files))
        else:
            data = np.memmap(self.path_catch_files + self.file_name + "-data(" + self.files_group + ").mymemmap",
                             dtype="uint8", mode="r+",
                             shape=(num_of_files, self.image_size * self.image_size * 3))
            labels = np.memmap(self.path_catch_files + self.file_name + "-labels(" + self.files_group + ").mymemmap",
                               dtype="int32", mode="r+",
                               shape=(num_of_files))

        downloaded_files = 0
        num_threads = th.active_count()
        while downloaded_files < num_of_files:

            if downloaded_files + num_threads > num_of_files:
                num_threads = num_of_files - downloaded_files

            threads = []
            self.thread_couple = []
            for i in range(num_threads):
                t = th.Thread(target=self.__transform_thread, args=(files[downloaded_files + i],))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            del threads

            for i in range(len(self.thread_couple)):
                data[downloaded_files + i] = self.thread_couple[i][0]
                labels[downloaded_files + i] = self.thread_couple[i][1]

            del self.thread_couple

            print(str(downloaded_files + num_threads) + "/" + str(num_of_files))

            downloaded_files += num_threads

        return

    # get function return img data and labels memory maps if they exist, or return None, None

    def get(self):
        if os.path.exists(self.path_catch_files + self.file_name + "-data(" + self.files_group + ").mymemmap"):
            num_of_files = len(self.__load_file_list())
            data = np.memmap(self.path_catch_files + self.file_name + "-data(" + self.files_group + ").mymemmap",
                             dtype="uint8", mode="r",
                             shape=(num_of_files, self.image_size * self.image_size * 3))
            labels = np.memmap(self.path_catch_files + self.file_name + "-labels(" + self.files_group + ").mymemmap",
                               dtype="int32", mode="r",
                               shape=(num_of_files))
            return data, labels
        else:
            print(
                "Fmow mymemmap file with name: " + self.file_name + "-data(" + self.files_group + ").mymemmap" + " dont exist!")
            return None, None

    # get function return normalized img data and labels memory maps if they exist, or return None, None

    def get_normalized(self):
        if os.path.exists(
                self.path_catch_files + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap"):
            num_of_files = len(self.__load_file_list())
            category_len = len(fmow_category_names)
            data = np.memmap(
                self.path_catch_files + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="r",
                shape=(num_of_files, self.image_size, self.image_size, 3))
            labels = np.memmap(
                self.path_catch_files + self.file_name + "-labels_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="r",
                shape=(num_of_files, category_len))
            return data, labels
        else:
            print(
                "Fmow mymemmap file with name: " + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap" + " dont exist!")
            return None, None

    # __normalized_thread function reshape img data and labels memory maps into normalized form

    def __normalized_thread(self, data_normalized, data, labels_normalized, labels, i):
        data_normalized[i] = np.reshape(data[i] / 255, (self.image_size, self.image_size, 3), order="F")
        labels_normalized[i][labels[i]] = 1

    # load_normalize creates normalized img data and labels memory maps
    # and fill them with reshaped data from non-normalized memory maps
    # through paralel threads using __normalized_thread function

    def load_normalize(self):
        files = self.__load_file_list()
        num_of_files = len(files)
        category_len = len(fmow_category_names)

        data, labels = self.get()

        if not os.path.exists(
                self.path_catch_files + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap"):
            data_normalized = np.memmap(
                self.path_catch_files + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="w+",
                shape=(num_of_files, self.image_size, self.image_size, 3))
            labels_normalized = np.memmap(
                self.path_catch_files + self.file_name + "-labels_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="w+",
                shape=(num_of_files, category_len))
        else:
            data_normalized = np.memmap(
                self.path_catch_files + self.file_name + "-data_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="r+",
                shape=(num_of_files, self.image_size, self.image_size, 3))
            labels_normalized = np.memmap(
                self.path_catch_files + self.file_name + "-labels_normalized(" + self.files_group + ").mymemmap",
                dtype="float32", mode="r+",
                shape=(num_of_files, category_len))

        normalized_files = 0
        num_threads = th.active_count()
        while normalized_files < num_of_files:

            if normalized_files + num_threads > num_of_files:
                num_threads = num_of_files - normalized_files

            threads = []
            for i in range(num_threads):
                t = th.Thread(target=self.__normalized_thread,
                              args=(data_normalized, data, labels_normalized, labels, normalized_files + i))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            del threads

            print(str(normalized_files + num_threads) + "/" + str(num_of_files))

            normalized_files += num_threads

        return


if __name__ == "__main__":
    print("Testing loading data:")

    # Generating
    dataset = Fmow("val_rgb")
    dataset.load()

    # Printing
    """
    data, labels = dataset.get()

    print(data)
    print(len(data))
    print(len(data[0]))
    print(data[0])
    print(type(data))
    print(data.dtype)
    print(labels)
    print(len(labels))
    print(type(labels))
    print(labels.dtype)
    """

    sys.exit()
