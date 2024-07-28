import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# 训练模型
def train_model(model, epochs, optimizer, criterion, source_data, source_label, batch_size, target_data, target_label,
                criterion_1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    source_data_tensor = torch.Tensor(np.asarray(source_data, dtype='float64'))
    source_label_tensor = torch.Tensor(np.asarray(source_label, dtype='float64'))
    target_data_tensor = torch.Tensor(np.asarray(target_data, dtype='float64'))
    target_label_tensor = torch.Tensor(np.asarray(target_label, dtype='float64'))

    source_data_tensor = torch.permute(source_data_tensor, (0, 2, 1))
    source_data_tensor = torch.reshape(source_data_tensor, (source_data_tensor.shape[0], 1,
                                                            source_data_tensor.shape[1],
                                                            source_data_tensor.shape[2]))
    target_data_tensor = torch.permute(target_data_tensor, (0, 2, 1))
    target_data_tensor = torch.reshape(target_data_tensor, (target_data_tensor.shape[0], 1,
                                                            target_data_tensor.shape[1],
                                                            target_data_tensor.shape[2]))

    source_dataset = TensorDataset(source_data_tensor, source_label_tensor)
    target_dataset = TensorDataset(target_data_tensor, target_label_tensor)

    source_data_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_data_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    train_step = 0
    writer = SummaryWriter('./Domain_Bi_LSTM_loss')

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0

        print('--------  Epoch {} begin  --------'.format(epoch + 1))

        for (source_x, source_y), (target_x, _) in tqdm(zip(source_data_loader, target_data_loader),
                                                        total=min(len(source_data_loader), len(target_data_loader))):
            source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)

            optimizer.zero_grad()

            p = float(train_step) / (epochs * len(source_data_loader))
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # RUL predict
            rul_pre, _ = model(source_x, grl_lambda)
            rul_loss = criterion(rul_pre, source_y)

            # domain classifier
            source_domain_labels = torch.zeros(len(source_x), dtype=torch.long, device=device)
            target_domain_labels = torch.ones(len(target_x), dtype=torch.long, device=device)
            domain_inputs = torch.cat((source_x, target_x), dim=0)
            domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)

            _, domain_pres = model(domain_inputs, grl_lambda)
            domain_loss = criterion_1(domain_pres, domain_labels)

            loss = rul_loss + domain_loss
            epoch_loss += loss.item()
            train_step += 1

            if train_step % 100 == 0:
                writer.add_scalar('train/rul_loss', rul_loss.item(), train_step)
                writer.add_scalar('train/domain_loss', domain_loss.item(), train_step)

            loss.backward()
            optimizer.step()

        print('-------- Epoch {} train_loss: {} --------'.format(epoch + 1, epoch_loss))
        writer.add_scalar('epoch_loss', epoch_loss, epoch + 1)

        if (epoch + 1) % 10 == 0:
            torch.save(model, f"./model/Domain_Bi_LSTM_epoch_{epoch + 1}.pth")

    torch.save(model, "./model/Domain_Bi_LSTM_final.pth")
    writer.close()
