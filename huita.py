import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class Dataset(data.Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.inputs = self.df.loc[:, ['Roman shifts', 'Counts', 'Brain region']].to_numpy(dtype=np.float16)
        self.outputs = self.df.loc[:, ['Result']].to_numpy(dtype=np.float16)
        del self.df

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        x = torch.tensor(self.inputs[item], dtype=torch.float16)
        y = torch.tensor(self.outputs[item], dtype=torch.float16)

        return x, y