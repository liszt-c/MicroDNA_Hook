import glob
import torch
from torch.utils import data
from torch.utils.data import _utils
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from dataprocess_transformer import dataprocess
import os

BATCH_SIZE = 128
Num_Workers = 4


###使用glob找到所有文件
all_DNA_path = glob.glob(r'./datasets/*/*.fa')
#for file in all_DNA_path:print(file)                      ###for test

###制作标签

all_labels = []
for DNA in all_DNA_path:
    if  'eccDNA' in DNA:
        all_labels.append((1,0))                ###v5修改标签为数组
    if  'otherDNA' in DNA: 
        all_labels.append((0,1))
all_labels = np.array(all_labels)               ###后续标签str类型报错
#print(all_labels)                             ###for test
#print(type(all_labels))

###继承数据集模型
class MyDataset(data.Dataset):
    def __init__(self, DNApath,labels):
        self.DNApath = DNApath
        self.labels = labels
    def __getitem__(self, index):
        DNA = self.DNApath[index]
        label = self.labels[index]
        DNA_seq = dataprocess(DNA)
        #data = torch.from_numpy(DNA_seq).float()        ###注意转化为float32
        data = np.array(DNA_seq, dtype=np.int64)
        label = np.array(label, dtype=np.int64)
        return data , label
    def __len__(self):
        return len(self.DNApath)

def dataloader(BATCH_SIZE):
    global all_DNA_path, all_labels
    DNA_datasets = MyDataset(all_DNA_path,all_labels)
    DNA_dataloader =  torch.utils.data.DataLoader(dataset=DNA_datasets,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers = Num_Workers)
    return DNA_dataloader


###划分训练集(80%)和验证集(20%)后调用
def data_for_run(BATCH_SIZE):
    global all_DNA_path, all_labels
    index = np.random.permutation(len(all_DNA_path))

    all_DNA_path = np.array(all_DNA_path)[index]
    all_labels = np.array(all_labels)[index]

    s = int(len(all_DNA_path)*0.8)
    #print(s)                               ###for test

    train_DNA = all_DNA_path[:s]
    train_labels = all_labels[:s]
    test_DNA = all_DNA_path[s:]
    test_labels = all_labels[s:]

    train_data = MyDataset(train_DNA, train_labels)
    test_data = MyDataset(test_DNA, test_labels)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers = Num_Workers,pin_memory = True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers = Num_Workers,pin_memory = True)

    return train_dataloader ,test_dataloader


###for test
if __name__ == "__main__":
    DNA_batch, labels_batch = next(iter(dataloader(BATCH_SIZE)))
    print(DNA_batch.shape)
    print(labels_batch.shape)