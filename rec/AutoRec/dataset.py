import torch
from torch.utils.data import Dataset


class AutoRecData(Dataset):
    def __init__(self,train, based_on):
        super(AutoRecData, self).__init__()
        self.train = train
        self.based_on = based_on
        self.n_user, self.n_item = train.shape

    def __len__(self):
        if self.based_on == 'item':
            return self.n_item
        elif self.based_on == 'user':
            return self.n_user
    
    def __getitem__(self, idx):
        if self.based_on == 'item':
            return torch.tensor(self.train[:, idx]).float()
        elif self.based_on == 'user':
            return torch.tensor(self.train[idx, :]).float()



