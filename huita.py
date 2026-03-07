import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3, 256),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 1024),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 3),
        )

    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        #out = self.layer5(out)
        return out


class Dataset(data.Dataset):
    def __init__(self, path, device):
        self.device = device
        self.df = pd.read_csv(path)
        self.inputs = torch.tensor(self.df.loc[:, ['Roman shifts', 'Counts', 'Brain region']].to_numpy(dtype=np.float32), dtype=torch.float32).to(self.device)
        self.outputs = self.df.loc[:, ['Result']].to_numpy(dtype=np.int16)
        del self.df

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        x = self.inputs[item]
        y = torch.tensor(np.eye(3)[self.outputs[item]], dtype=torch.float32)

        return x, y