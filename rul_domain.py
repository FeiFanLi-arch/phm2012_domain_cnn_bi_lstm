import random
import time

import numpy as np
import torch
from torch import nn

from project.domain_train import train_model
from project.rul_pre_data import get_train_data, get_test_data


# load model
def load_model():
    base_model = torch.load('/project/model/baseline.pth')
    domain_model = fine_tuning_model(base_model)

    return domain_model


def fine_tuning_model(raw_model):
    set_parameter_requires_grad(raw_model)
    raw_model.get_time_feature = nn.LSTM(160, 64, 2, dropout=0.5, batch_first=True, bidirectional=True)
    raw_model.fc = nn.Conv1d(1, 1, 128, 128)
    raw_model.domain_classifier = nn.Sequential(
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 2)
    )

    return raw_model


def set_parameter_requires_grad(new_model):
    for param in new_model.get_feature.parameters():
        param.requires_grad = False


# deal data
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

model = load_model()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = nn.MSELoss()
criterion_1 = nn.CrossEntropyLoss()
data_0, label_0, data_1, label_1, data_2, label_2, data_3, label_3, data_4, label_4, data_5, label_5 = get_train_data()
data_6, label_6, data_7, label_7, data_8, label_8, data_9, label_9, data_10, label_10, data_11, label_11, \
    data_12, label_12, data_13, label_13, data_14, label_14, data_15, label_15, data_16, label_16 = get_test_data()
s_data_0, s_label_0 = deal_dataset(data_0, label_0, 32)
s_data_1, s_label_1 = deal_dataset(data_1, label_1, 32)
t_data_2, t_label_2 = deal_dataset(data_2, label_2, 32)
t_data_3, t_label_3 = deal_dataset(data_3, label_3, 32)
t_data_4, t_label_4 = deal_dataset(data_4, label_4, 32)
t_data_5, t_label_5 = deal_dataset(data_5, label_5, 32)

s_dataset = s_data_0 + s_data_1
s_labels = s_label_0 + s_label_1
t_dataset = t_data_2 + t_data_3 + t_data_4 + t_data_5
t_labels = t_label_2 + t_label_3 + t_label_4 + t_label_5

if __name__ == '__main__':
    epochs = 200

    start_time = time.perf_counter()
    train_model(model, epochs, optimizer, criterion, s_dataset, s_labels, 32, t_dataset, t_labels, criterion_1)
    end_time = time.perf_counter()
    final_time = end_time - start_time
    print(f"The elapsed time is {final_time} seconds.")
