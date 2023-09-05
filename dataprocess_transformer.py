import numpy
import random

norm_length = 400       ###V7归一化长度

def dataprocess(file_name):
    ###read DNA seq
    with open(file_name, "r", encoding='utf-8') as DNA_seq:        # file_name transfer from the training code
        seqs = DNA_seq.readlines()
    ###Skip comment lines
    seq_clear=''
    for seq in seqs:
        if not seq.strip().startswith(">"):
            seq_clear = seq_clear + seq   
    ###Remove Spaces and newlines
    seq_clear.replace(" ","")
    seq_clear.replace("\n","")
    ###turn to the vector
    n = len(seq_clear)-1
    #print(seq_clear)                                               #for test
    if n <=norm_length:
        data = numpy.zeros((norm_length), dtype = int)
        fund = int((norm_length - n)/2 )
        for j in range(fund):
            x = random.randint(1,4)
            local = x % 4
            data[j] = local
        for i in range(n):
            if (seq_clear[i]=='A' or seq_clear[i]=='a'):
                data[fund+i]  = 0
            if (seq_clear[i]=='T' or seq_clear[i]=='t'):
                data[fund+i]  = 1
            if (seq_clear[i]=='C' or seq_clear[i]=='c'):
                data[fund+i]  = 2
            if (seq_clear[i]=='G' or seq_clear[i]=='g'):
                data[fund+i]  = 3
        for ww in range(norm_length - n - fund):
            x = random.randint(1,4)
            local = x % 4
            data[fund+n+ww] = local
    if n >norm_length:
        data = numpy.zeros((norm_length), dtype = int)
        fund = int((n - norm_length)/2)
        for i in range(norm_length):
            if (seq_clear[i+fund]=='A' or seq_clear[i+fund]=='a'):
                data[i]  = 0
            if (seq_clear[i+fund]=='T' or seq_clear[i+fund]=='t'):
                data[i]  = 1
            if (seq_clear[i+fund]=='C' or seq_clear[i+fund]=='c'):
                data[i]  = 2
            if (seq_clear[i+fund]=='G' or seq_clear[i+fund]=='g'):
                data[i]  = 3
    return data


if __name__ == '__main__':                                          #for test
    data = dataprocess('./datasets/eccDNA/chr50.fa')
    numpy.set_printoptions(threshold=numpy.inf)
    print(data)
    print(data.shape)