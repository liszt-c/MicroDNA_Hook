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
        data = numpy.zeros((4,norm_length), dtype = int)
        fund = int((norm_length - n)/2 )
        for j in range(fund):
            x = random.randint(1,4)
            local = x % 4
            data[local,j] = 1
        for i in range(n):
            if (seq_clear[i]=='A' or seq_clear[i]=='a'):
                data[0,fund+i]  = 1
            if (seq_clear[i]=='T' or seq_clear[i]=='t'):
                data[1,fund+i]  = 1
            if (seq_clear[i]=='C' or seq_clear[i]=='c'):
                data[2,fund+i]  = 1
            if (seq_clear[i]=='G' or seq_clear[i]=='g'):
                data[3,fund+i]  = 1
        for ww in range(norm_length - n - fund):
            x = random.randint(1,4)
            local = x % 4
            data[local,fund+n+ww] = 1
    if n >norm_length:
        data = numpy.zeros((4,norm_length), dtype = int)
        fund = int((n - norm_length)/2)
        for i in range(norm_length):
            if (seq_clear[i+fund]=='A' or seq_clear[i+fund]=='a'):
                data[0,i]  = 1
            if (seq_clear[i+fund]=='T' or seq_clear[i+fund]=='t'):
                data[1,i]  = 1
            if (seq_clear[i+fund]=='C' or seq_clear[i+fund]=='c'):
                data[2,i]  = 1
            if (seq_clear[i+fund]=='G' or seq_clear[i+fund]=='g'):
                data[3,i]  = 1
    return data


if __name__ == '__main__':                                          #for test
    data = dataprocess('chr50.fa')
    numpy.set_printoptions(threshold=numpy.inf)
    print(data)
    print(data.shape)