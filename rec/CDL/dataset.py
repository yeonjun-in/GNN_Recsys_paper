import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os, sys
#find vocabulary_size = 8000

# os.chdir('/home/yeonjun/Desktop/RecSys_implementation/CDL')
# sys.path.append(os.getcwd())


# with open(r"./data/vocabulary.dat") as vocabulary_file:
#     vocabulary_size = len(vocabulary_file.readlines())
    
# #find item_size = 16980
# with open(r"./data/mult.dat") as item_info_file:
#     item_size = len(item_info_file.readlines())

# #initialize item_infomation_matrix (16980 , 8000)
# item_infomation_matrix = np.zeros((item_size , vocabulary_size))

# #build item_infomation_matrix
# with open(r"./data/mult.dat") as item_info_file:
#     sentences = item_info_file.readlines()
    
#     for index,sentence in enumerate(sentences):
#         words = sentence.strip().split(" ")[1:]
#         for word in words:
#             vocabulary_index , number = word.split(":")
#             item_infomation_matrix[index][int(vocabulary_index)] =number

# #find user_size = 5551
# with open(r"./data/users.dat") as rating_file:
#     user_size = len(rating_file.readlines())

# #initialize rating_matrix (5551 , 16980)
# import numpy as np
# rating_matrix = np.zeros((user_size , item_size))

# #build rating_matrix
# with open(r"./data/users.dat") as rating_file:
#     lines = rating_file.readlines()
#     for index,line in enumerate(lines):
#         items = line.strip().split(" ")
#         for item in items:  
#             rating_matrix[index][int(item)] = 1

def add_noise(x, corrupt_ratio):
    noise = np.random.binomial(1, corrupt_ratio, size=x.shape)
    return x + noise

class CDLDataset(Dataset):
    def __init__(self, xc, x0):
        super(CDLDataset, self).__init__()
        self.xc = xc
        self.x0 = x0
    
    def __len__(self):
        return self.xc.shape[0]

    def __getitem__(self, idx):
        return {'clean':torch.FloatTensor(self.xc[idx, :]),
                'corrupt':torch.FloatTensor(self.x0[idx, :]),
                'idx':idx}
