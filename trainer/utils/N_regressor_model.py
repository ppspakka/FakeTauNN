# a pytorch model for learning the number of fakes in an event from the pileup
import torch.nn as nn
import torch.nn.functional as F


class N_regressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout=0.2):
        super(N_regressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = self.batchnorm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout)
        x = self.fc3(x)
        return x
    