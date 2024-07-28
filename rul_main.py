import random
import time
import warnings

import numpy as np
import torch.optim
from torch import nn

from project.rul_model import LSTM
from project.rul_pre_data import get_train_data, get_test_data
from project.rul_train import train_model

warnings.filterwarnings('ignore')


def deal_dataset(data, label, length):
    dataset = []
    labels = []
    for i in range(len(data)):
        data_length = int(len(data[i]) / length) * length
        for j in range(data_length):
            dataset.append(data[i][j])
            labels.append(label[i][j])

    return dataset, labels


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

input_size = 160
hidden_size = 64
num_layers = 2
batch_size = 32

model = LSTM(input_size, hidden_size, num_layers, 'relu')

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

criterion = nn.MSELoss()
criterion_1 = nn.CrossEntropyLoss()

data_0, label_0, data_1, label_1, data_2, label_2, data_3, label_3, data_4, label_4, data_5, label_5 = get_train_data()

train_data_0, train_label_0 = deal_dataset(data_0, label_0, batch_size)
train_data_1, train_label_1 = deal_dataset(data_1, label_1, batch_size)
train_data_2, train_label_2 = deal_dataset(data_2, label_2, batch_size)
train_data_3, train_label_3 = deal_dataset(data_3, label_3, batch_size)
train_data_4, train_label_4 = deal_dataset(data_4, label_4, batch_size)
train_data_5, train_label_5 = deal_dataset(data_5, label_5, batch_size)

data_6, label_6, data_7, label_7, data_8, label_8, data_9, label_9, data_10, label_10, data_11, label_11, data_12, \
    label_12, data_13, label_13, data_14, label_14, data_15, label_15, data_16, label_16 = get_test_data()

test_data_1, test_label_1 = deal_dataset(data_6, label_6, batch_size)
test_data_2, test_label_2 = deal_dataset(data_7, label_7, batch_size)
test_data_3, test_label_3 = deal_dataset(data_8, label_8, batch_size)
test_data_4, test_label_4 = deal_dataset(data_9, label_9, batch_size)
test_data_5, test_label_5 = deal_dataset(data_10, label_10, batch_size)

train_dataset = train_data_0 + train_data_1
train_labels = train_label_0 + train_label_1
test_dataset = train_data_2 + train_data_3 + train_data_4 + train_data_5
test_labels = train_label_2 + train_label_3 + train_label_4 + train_label_5

if __name__ == '__main__':
    train_epochs = 200

    start_time = time.perf_counter()
    train_model(model, train_epochs, optimizer, criterion, train_dataset, train_labels, batch_size, test_dataset,
                test_labels, criterion_1)
    end_time = time.perf_counter()
    final_time = end_time - start_time
    print(f"The elapsed time is {final_time} seconds.")
