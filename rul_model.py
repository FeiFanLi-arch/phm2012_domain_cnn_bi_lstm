import torch
from torch import nn
from torch.autograd import Function


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha

        return grad_output, None


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_function):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        elif activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        self.get_feature = nn.Sequential(
            nn.Conv2d(1, 16, (1, 8), padding=4),
            self.activation_function,
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4),

            nn.Conv2d(16, 16, (1, 4), padding=4),
            self.activation_function,
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4),

            nn.Conv2d(16, 16, (1, 4), padding=4),
            self.activation_function,
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4)

        )

        self.get_time_feature = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Conv1d(1, 1, 2*self.hidden_size, 2*self.hidden_size)
        )

    # def forward(self, x):
    #     feature = self.get_feature(x)
    #     feature = feature.view(-1, 160)
    #     time_feature, _ = self.get_time_feature(feature)
    #     rul_pre = self.fc(time_feature.view(-1, 1, 2*self.hidden_size))
    #
    #     return rul_pre.view(-1)

    def forward(self, x, grl_lambda=None):
        feature = self.get_feature(x)
        feature = feature.view(-1, 160)
        time_feature, _ = self.get_time_feature(feature)

        rul_pre = self.fc(time_feature.view(-1, 1, 2*self.hidden_size))

        reverse_feature = GradReverse.apply(time_feature.view(time_feature.size(0), -1), grl_lambda)
        domain_pre = self.domain_classifier(reverse_feature)

        return rul_pre.view(-1), domain_pre
