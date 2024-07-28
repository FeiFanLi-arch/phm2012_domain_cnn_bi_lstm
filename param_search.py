import random
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from project.rul_model import LSTM
from project.rul_pre_data import get_train_data, get_test_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# train model
def train_model(t_model, epochs, optimizer, criterion, data, label, batch_size, data_1, label_1):
    train_data_tensor = torch.Tensor(np.asarray(data, dtype='float64')).to(device)
    train_label_tensor = torch.Tensor(np.asarray(label, dtype='float64')).to(device)
    test_data_tensor = torch.Tensor(np.asarray(data_1, dtype='float64')).to(device)
    test_label_tensor = torch.Tensor(np.asarray(label_1, dtype='float64')).to(device)

    train_data_tensor_p = torch.permute(train_data_tensor, (0, 2, 1))
    test_data_tensor_p = torch.permute(test_data_tensor, (0, 2, 1))

    train_data_final = torch.reshape(train_data_tensor_p, (train_data_tensor_p.shape[0], 1, train_data_tensor_p.shape[1]
                                                           , train_data_tensor_p.shape[2]))
    test_data_final = torch.reshape(test_data_tensor_p, (test_data_tensor_p.shape[0], 1, test_data_tensor_p.shape[1],
                                                         test_data_tensor_p.shape[2]))
    train_data_loader = DataLoader(train_data_final, batch_size=batch_size)
    train_label_loader = DataLoader(train_label_tensor, batch_size=batch_size)

    t_model.train()
    for i in range(epochs):
        for data_load in tqdm(zip(train_data_loader, train_label_loader), total=len(train_data_loader)):
            optimizer.zero_grad()
            output = t_model(data_load[0].to(device))
            loss = criterion(output, data_load[1].to(device))
            loss.backward()
            optimizer.step()

    t_model.eval()
    with torch.no_grad():
        predict = t_model(test_data_final)
        predict_loss = np.sqrt(mean_squared_error(test_label_tensor.cpu().numpy(), predict.cpu().numpy()))

    return predict_loss


param_grid = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'activation_function': ['relu', 'leaky_relu', 'tanh']
}

data_0, label_0, data_1, label_1, data_2, label_2, data_3, label_3, data_4, label_4, data_5, label_5 = get_train_data()
data_6, label_6, data_7, label_7, data_8, label_8, data_9, label_9, data_10, label_10, data_11, label_11, data_12, \
    label_12, data_13, label_13, data_14, label_14, data_15, label_15, data_16, label_16 = get_test_data()

train_data_0, train_label_0 = deal_dataset(data_0, label_0, 1)
train_data_1, train_label_1 = deal_dataset(data_1, label_1, 1)
train_dataset = train_data_0 + train_data_1
train_labels = train_label_0 + train_label_1
test_data_1, test_label_1 = deal_dataset(data_6, label_6, 1)
test_data_2, test_label_2 = deal_dataset(data_7, label_7, 1)
test_data_3, test_label_3 = deal_dataset(data_8, label_8, 1)
test_data_4, test_label_4 = deal_dataset(data_9, label_9, 1)
test_data_5, test_label_5 = deal_dataset(data_10, label_10, 1)
test_dataset = test_data_1 + test_data_2 + test_data_3 + test_data_4 + test_data_5
test_labels = test_label_1 + test_label_2 + test_label_3 + test_label_4 + test_label_5

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

best_params = None
best_score = float('inf')
n_iter = 10

for _ in range(n_iter):
    param_dict = {key: random.choice(values) for key, values in param_grid.items()}
    model = LSTM(
        input_size=160,
        hidden_size=param_dict['hidden_size'],
        num_layers=param_dict['num_layers'],
        activation_function=param_dict['activation_function']
    ).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
    batch_size = param_dict['batch_size']

    try:
        score = train_model(model, 10, optimizer, criterion, train_dataset, train_labels, batch_size, test_dataset, test_labels)
        if score < best_score:
            best_score = score
            best_params = param_dict
        print(f'Parameters: {param_dict}, RMSE: {score}')
    except Exception as e:
        print(f'Error with parameters {param_dict}: {e}')

print(f'Final best parameters: {best_params}')
print(f'Final best RMSE: {best_score}')
