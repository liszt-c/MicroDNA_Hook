import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import scipy.io as sio


BATCH_SIZE = 10
Num_Workers = 0


filename2 = './datasets/test.mat'
data = sio.loadmat(filename2)
test_DNAs = data['testxdata'] # shape = (455024, 4, 1000)
test_LABELs = data['testdata'] # shape = (455024, 919)
#print(type(test_LABELs))
#<class 'numpy.ndarray'>
#test_DNAs = np.transpose(test_DNAs, (0, 1, 2)) 
#test_LABELs = np.transpose(test_LABELs, (0, 1)) 
print ('test_Data shape:',test_DNAs.shape,test_LABELs.shape)
print('test_Data loaded!')
length = len(test_DNAs)
print('length:',length)


###shap must be (batchsize,chanel,data)

filename = './datasets/train.mat'
with h5py.File(filename, 'r') as file:
    DNAs = file['trainxdata']               # shape = (1000, 4, 4400000)
    LABELs = file['traindata']              # shape = (919, 4400000)
    DNAs = np.transpose(DNAs, (2, 1, 0))    
    LABELs = np.transpose(LABELs, (1, 0))   
print ('Data shape:',DNAs.shape,LABELs.shape)
print('Data loaded!')
length = np.size(DNAs,0)
print('length:',length)


###继承数据集模型
class MyDataset(Dataset):
    def __init__(self, DNAlist,labels):
        self.DNAlist = DNAlist
        self.labels = labels
    def __getitem__(self, index):
        DNAs = self.DNAlist[index]
        label = self.labels[index]
        DNAs = torch.from_numpy(DNAs).float()
        label = torch.from_numpy(label).float()
        return DNAs , label
    def __len__(self):
        return len(self.DNAlist)

def dataloader(BATCH_SIZE):
    global test_DNAs,test_LABELs
    DNA_datasets = MyDataset(test_DNAs,test_LABELs)
    DNA_dataloader =  torch.utils.data.DataLoader(dataset=DNA_datasets,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers = Num_Workers)
    return DNA_dataloader


def data_for_run(BATCH_SIZE):
    global DNAs, LABELs, test_DNAs, test_LABELs

    train_data = MyDataset(DNAs, LABELs)
    test_data = MyDataset(test_DNAs, test_LABELs)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader ,test_dataloader


###for test
if __name__ == "__main__":
    DNA_batch, labels_batch = next(iter(dataloader(BATCH_SIZE)))
    print(DNA_batch.shape)
    print(labels_batch.shape)