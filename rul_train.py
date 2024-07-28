import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_model(model, epochs, optimizer, criterion, data, label, batch_size, data_1, label_1, criterion_1):
    train_data_tensor = torch.Tensor(np.asarray(data, dtype='float64'))
    train_label_tensor = torch.Tensor(np.asarray(label, dtype='float64'))
    test_data_tensor = torch.Tensor(np.asarray(data_1, dtype='float64'))
    test_label_tensor = torch.Tensor(np.asarray(label_1, dtype='float64'))

    train_data_tensor_p = torch.permute(train_data_tensor, (0, 2, 1))
    test_data_tensor_p = torch.permute(test_data_tensor, (0, 2, 1))

    train_data_final = torch.reshape(train_data_tensor_p, (train_data_tensor_p.shape[0], 1, train_data_tensor_p.shape[1]
                                                           , train_data_tensor_p.shape[2]))

    train_data_loader = DataLoader(train_data_final, batch_size=batch_size)
    train_label_loader = DataLoader(train_label_tensor, batch_size=batch_size)

    train_step = 0
    writer_1 = SummaryWriter('./Bi-LSTM_baseline-loss')

    for i in range(epochs):

        epoch_loss = 0
        model.train()

        print('--------epoch {} begin--------'.format(i+1))

        for data_load in tqdm(zip(train_data_loader, train_label_loader), total=len(train_data_loader)):
            optimizer.zero_grad()
            output = model(data_load[0])
            loss = criterion(output, data_load[1])
            epoch_loss += loss.item()
            train_step += 1
            if train_step % 100 == 0:
                writer_1.add_scalar('loss', loss.item(), train_step)
            loss.backward()
            optimizer.step()

        print('')
        print('--------epoch {} total_loss：{}'.format(i + 1, epoch_loss))
        print('--------epoch {} end--------'.format(i + 1))
        writer_1.add_scalar('epoch_loss', epoch_loss, i + 1)
        if i + 1 == epochs:
            torch.save(model, "./model/Bi-LSTM_baseline.pth")

    writer_1.close()
