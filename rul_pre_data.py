import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalization_data(data):
    data = np.array(data.iloc[:, 4:])
    data = MinMaxScaler().fit_transform(data)

    return data


def normalization_label(label):
    labels = []
    for i in range(len(label)):
        new_label = (label[i] - min(label)) / (max(label) - min(label))
        labels.append(new_label)

    return labels


def normalization_test_label(label, max_rul):
    labels = []
    for i in range(len(label)):
        new_label = label[i] / max_rul
        labels.append(new_label)

    return labels


def split_data(data, label):
    num = int(len(data[0]) / 256)
    dataset_1 = []
    dataset_2 = []
    dataset_3 = []
    dataset_4 = []
    dataset_5 = []
    dataset_6 = []
    dataset_7 = []
    dataset_8 = []
    dataset_9 = []
    dataset_10 = []
    dataset = []
    labels = []
    for i in range(len(data)):
        for j in range(num):
            sub_data = data[i][j * 256:(j + 1) * 256]
            if j == 0:
                dataset_1.append(sub_data)
            elif j == 1:
                dataset_2.append(sub_data)
            elif j == 2:
                dataset_3.append(sub_data)
            elif j == 3:
                dataset_4.append(sub_data)
            elif j == 4:
                dataset_5.append(sub_data)
            elif j == 5:
                dataset_6.append(sub_data)
            elif j == 6:
                dataset_7.append(sub_data)
            elif j == 7:
                dataset_8.append(sub_data)
            elif j == 8:
                dataset_9.append(sub_data)
            else:
                dataset_10.append(sub_data)

    dataset.append(dataset_1)
    dataset.append(dataset_2)
    dataset.append(dataset_3)
    dataset.append(dataset_4)
    dataset.append(dataset_5)
    dataset.append(dataset_6)
    dataset.append(dataset_7)
    dataset.append(dataset_8)
    dataset.append(dataset_9)
    dataset.append(dataset_10)

    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)
    labels.append(label)

    return dataset, labels


def load_data(root, path):
    paths = root + path

    filenames = os.listdir(paths)

    datasets = []
    labels = []

    for i, filename in enumerate(filenames, 0):
        sub_path = os.path.join(paths, filename)
        data = pd.read_csv(sub_path, header=None)
        datas = normalization_data(data)
        datasets.append(datas)
        label = (len(filenames) - i) * 10
        labels.append(label)

    labels = normalization_label(labels)

    return split_data(datasets, labels)

    # return datasets, labels


def load_test_data(root, path, rul):
    paths = root + path

    filenames = os.listdir(paths)

    datasets = []
    labels = []
    max_rul = rul + (len(filenames))

    for i, filename in enumerate(filenames, 0):
        sub_path = os.path.join(paths, filename)
        data = pd.read_csv(sub_path, header=None)
        datas = normalization_data(data)
        datasets.append(datas)
        label = max_rul - i
        labels.append(label)

    labels = normalization_test_label(labels, max_rul)

    return split_data(datasets, labels)

    # return datasets, labels


def get_train_data():
    root = './dataset_1/Learning_set/'
    path_1 = 'Bearing1_1'
    path_2 = 'Bearing1_2'
    path_3 = 'Bearing2_1'
    path_4 = 'Bearing2_2'
    path_5 = 'Bearing3_1'
    path_6 = 'Bearing3_2'
    dataset_1, label_1 = load_data(root, path_1)
    dataset_2, label_2 = load_data(root, path_2)
    dataset_3, label_3 = load_data(root, path_3)
    dataset_4, label_4 = load_data(root, path_4)
    dataset_5, label_5 = load_data(root, path_5)
    dataset_6, label_6 = load_data(root, path_6)

    return dataset_1, label_1, dataset_2, label_2, dataset_3, label_3, dataset_4, label_4, dataset_5, label_5, dataset_6, label_6


def get_test_data():
    root = './dataset_1/Full_Test_Set/'
    path_1 = 'Bearing1_3'
    path_2 = 'Bearing1_4'
    path_3 = 'Bearing1_5'
    path_4 = 'Bearing1_6'
    path_5 = 'Bearing1_7'
    path_6 = 'Bearing2_3'
    path_7 = 'Bearing2_4'
    path_8 = 'Bearing2_5'
    path_9 = 'Bearing2_6'
    path_10 = 'Bearing2_7'
    path_11 = 'Bearing3_3'

    dataset_1, label_1 = load_test_data(root, path_1, 573)
    dataset_2, label_2 = load_test_data(root, path_2, 34)
    dataset_3, label_3 = load_test_data(root, path_3, 161)
    dataset_4, label_4 = load_test_data(root, path_4, 146)
    dataset_5, label_5 = load_test_data(root, path_5, 757)
    dataset_6, label_6 = load_test_data(root, path_6, 753)
    dataset_7, label_7 = load_test_data(root, path_7, 139)
    dataset_8, label_8 = load_test_data(root, path_8, 309)
    dataset_9, label_9 = load_test_data(root, path_9, 129)
    dataset_10, label_10 = load_test_data(root, path_10, 58)
    dataset_11, label_11 = load_test_data(root, path_11, 82)

    return dataset_1, label_1, dataset_2, label_2, dataset_3, label_3, dataset_4, label_4, dataset_5, label_5, \
           dataset_6, label_6, dataset_7, label_7, dataset_8, label_8, dataset_9, label_9, dataset_10, label_10, \
           dataset_11, label_11
