import torch.nn as nn
import torch
from torch.nn import functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = []
        self.relu = nn.ReLU(inplace=True)
        self.pool_layers = []
        self.norm = []
        # Information for the layers
        kernel_size = [(5, 5), (5, 5), (3, 3), (3, 3), (3, 3)]
        feature_values = [1, 32, 64, 128, 128, 256]
        stride_values = [(2, 2), (2, 2), (1, 2), (1, 2), (1,2)]

        num_of_layers = len(stride_values)

        # Creating the layers of the CNN network
        for i in range(num_of_layers):
            self.conv_layers.append(nn.Conv2d(feature_values[i], feature_values[i + 1],
                kernel_size=kernel_size[i], stride=(1, 1), padding="same"))
            self.norm.append(nn.BatchNorm2d(feature_values[i + 1]))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=stride_values[i], stride=stride_values[i], padding=(0, 0)))

        # Creating a LSTM
        # batch_first = True, the input and output tensor are formatted as (batch, seq, feature)
        hidden_size = 256
        self.lstm = nn.LSTM(input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

        self.pre_ctc_cnn = nn.Conv2d(hidden_size*2, 80, kernel_size=(1,1), stride=(1, 1), padding="same")

    def forward(self, img):
        x = img
        for i in range(len(self.pool_layers)):
            x = self.conv_layers[i](x)
            x = self.norm[i](x)
            x = self.relu(x)
            x = self.pool_layers[i](x)

        x = torch.reshape(x, (1, 32, 256))

        x = self.lstm(x)[0]

        #BxTx2H -> BxTx1X2H
        x = torch.unsqueeze(x, dim=2)
        # project output to chars (including blank): BxTx1x2H -> Bx2Hx1xT -> BxCx1xT -> BxCxT -> BxTxC
        x = torch.permute(x, (0, 3, 2, 1))
        x = self.pre_ctc_cnn(x)
        x = torch.squeeze(x, 2)
        x = torch.permute(x, (0, 2, 1))

        x = torch.permute(x, (1, 0, 2))  # BxTxC -> TxBxC
        log_probs = F.log_softmax(x, 2)

        return log_probs