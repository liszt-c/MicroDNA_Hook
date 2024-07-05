'''
###需求
1.输入原始测序文件
文件转化路径：SRR-fastq-bam-callpeaks
SAMtools-bowtie2-MACS2
设置识别阈值，strict，normal，relax

2.对于长序列文件例如fasta文件，注意跳过>

3.手动输入序列

###结构

1.识别核
2.在输入中滑动窗口，每次调用识别核
3.检测输入长度，大于400bp调用滑动窗口，小于400bp使用补充噪音(直接调用datapeocess)直接输出检测结果
4.长片段得出的是(400-10)/10长度的数据，每40个一组再次滑动窗口找到窗口和大于20的区域，越大可能性越高
5.callpeaks结果的提取与预测，将序列位置，预测结果写入表格中

'''
import argparse
import torch
import torchvision
import os
import sys
import numpy
import random
import glob
import torch.nn.functional as nn
torch.set_printoptions(profile="full")
#from ResNet_Attention import ResNetAttention
from ResAttention import ResNetSelfAttention
import datetime

# Slide the window for the first time
norm_length = 400
STEP1 = 10
# Slide the window for the second time
WINDOW2 = 10
STEP2 = 3


time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)

def dataprocess(seq_clear):
    n = len(seq_clear)-1
    if n <norm_length:                                     ###前后噪音填充
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
    if n >= norm_length:
        data = numpy.zeros((4,n), dtype = int)
        for i in range(n):
            if (seq_clear[i]=='A' or seq_clear[i]=='a'):
                data[0,i]  = 1
            if (seq_clear[i]=='T' or seq_clear[i]=='t'):
                data[1,i]  = 1
            if (seq_clear[i]=='C' or seq_clear[i]=='c'):
                data[2,i]  = 1
            if (seq_clear[i]=='G' or seq_clear[i]=='g'):
                data[3,i]  = 1
    return data,n


def dataprocess2(file_name):
    #print(file_name)
    with open(file_name, "r", encoding='utf-8') as DNA_seq:
        seqs = DNA_seq.readlines()
    ###Skip comment lines
    seq_clear=''
    for seq in seqs:
        #print(seq)
        if not seq.strip().startswith(">"):
            seq_clear = seq_clear + seq   
    ###Remove Spaces and newlines
    seq_clear.replace(" ","")
    seq_clear.replace("\n","")
    #print(seq_clear)
    data,n = dataprocess(seq_clear)
    return data,n,seq_clear



def Recognition_kernel(input_seq,model): 
    ###dataprocess输出是(4,400)，需要变为(1,4,400),torch.unsqueeze用于升维
    input_seq = torch.unsqueeze(input_seq, dim=0)
    #print(input_seq)
    if torch.cuda.is_available():
        input_seq = input_seq.cuda()
    outputs = model(input_seq)
    #print(outputs)
    result = torch.nn.functional.softmax(outputs,dim = -1)
    #print(result,'\n')
    ###(1,0)是eccDNA,(0,1)是otherDNA
    A = result[0,0].item()
    B = result[0,1].item()
    return A-B,A



def forecast(DNA_matrix,n,model):
    if (n<=400):
        inpute = torch.from_numpy(numpy.asarray(DNA_matrix)).float()
        result,prob = Recognition_kernel(inpute,model)
    if(n>400):              ###以20bp为窗口滑动,最后一个窗口滑动距离小于20bp
        sum = 0
        inpute = DNA_matrix[:,n-400:n]
        inpute = torch.from_numpy(numpy.asarray(inpute)).float()
        outputs,probability = Recognition_kernel(inpute,model)
        max = outputs
        prob = probability
        if(n-400>20):
            for i in range(int((n-400)/20)):    #int向下取整
                inpute = DNA_matrix[:,(i*20):(400+i*20)]
                inpute = torch.from_numpy(numpy.asarray(inpute)).float()
                outputs,probability = Recognition_kernel(inpute,model)
                sum = sum + outputs
                max = outputs
                prob = probability
                if(outputs>max):
                    max = outputs
                    prob = probability
        result = max
    return result,prob



def fasta_base(file_name,model):
    DNA_matrix,n,_ = dataprocess2(file_name)      #(4,n)矩阵
    result,prob = forecast(DNA_matrix,n,model)
    return result,prob


def manual_base(seq,model):
    DNA_matrix,n = dataprocess(seq)      #(4,n)矩阵
    result,prob = forecast(DNA_matrix,n,model)
    return result,prob




def long_segment(file_name,model,limit):
    limit = float(limit)
    DNA_matrix,n,seq_clear = dataprocess2(file_name)
    #print(DNA_matrix.shape)
    print('输入长度：',n)
    if(n<800):
        print('Length less than 800bp, short sequence identification is better!')
        return(0,0,0,0)
    ###400bp长度，10bp步长滑动窗口
    sliding_block = [0]*int((n-400)/STEP1+2)
    for i in range(int((n-400)/STEP1)+1):        ###int向下取整，例如956bp，分为55份，最后需要对6再次进行滑动，滑动56次
        inpute = DNA_matrix[:,(i*STEP1):(400+i*STEP1)]        ###numpy切片右边是开去区间，天然可以避免越界
        inpute = torch.from_numpy(numpy.asarray(inpute)).float()
        outputs,probability = Recognition_kernel(inpute,model)
        sliding_block[i] = probability
    ##修改检测末尾
    if((n-400)%STEP1 != 0):
        inpute = DNA_matrix[:,(n-400):(n)]
        inpute = torch.from_numpy(numpy.asarray(inpute)).float()
        outputs,probability = Recognition_kernel(inpute,model)
        sliding_block[i+1] = probability
    ##无BUG已检测 v10

        #print(sliding_block)
    ###得到滑动结果后，再次滑动窗口，以10bp窗口、3bp步长滑动
    block_lenth = len(sliding_block)
    sliding_block = numpy.array(sliding_block)  ###numpy可调用sum方法
    print('block_lenth lenth:',block_lenth)
    summ1 = [0]*int((block_lenth-WINDOW2)/STEP2+2)
    for j in range(int((block_lenth-WINDOW2)/STEP2+1)):
        summ1[j] = (sliding_block[j*STEP2:(WINDOW2+j*STEP2)].sum())/WINDOW2
    if((block_lenth-WINDOW2)%STEP2 != 0):
        summ1[j+1] = (sliding_block[j*STEP2:block_lenth].sum())/(block_lenth-j*STEP2)
    ##无BUG已检测 v10


    block_lenth2 = len(summ1)
    summ2 = [-1]*int(block_lenth2)
    summ3 = [-1]*int(block_lenth2)
    summ2 = numpy.array(summ2)
    summ3 = numpy.array(summ3)
    ###左右区间计数列表申请较大，到时候需比较
    left = [-1]*int(block_lenth2/STEP2)
    right = [-1]*int(block_lenth2/STEP2)
    print('block_lenth2 lenth:',block_lenth2)
    print('limit:',limit)

    #print(summ1)
    #for循环计数对矩阵重写，大于limit记为1，小于记为0
    for xxx in range(block_lenth2):
        if(summ1[xxx] >= limit):
            summ2[xxx] = 1
        if(summ1[xxx] < limit):
            summ2[xxx] = 0
    #以2为窗口，1为步长滑动窗口。
    for j in range(block_lenth2-1):
        summ3[j] = summ2[j:(2+j)].sum() #值可能为0,1,2

    #print(summ3)
    #找到第一个为正的值，记入左矩阵，之后找到第一个
    kkk = 0
    for sss in range(block_lenth2-1):
        if(summ3[sss]==2):
            if(left[kkk]==-1):  ###左值没有填充，所以将当前序号填充左值
                left[kkk]=sss
        if(summ3[sss]==0):
            if(left[kkk]!=-1):  ###左值填充过了，右值没有填充，且检测当前值为0，将当前序号填入右值并进入下一个左右值
                if(right[kkk]==-1):
                    right[kkk]=sss
                    kkk = kkk + 1
    if(left[kkk]!=-1 and right[kkk]==-1):   ###对于最后一个左值，没有填充右值，则将最后值填充进入
        right[kkk] = block_lenth2 -1

    #print(left,'/n',right)

    return(left,right,kkk,seq_clear)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'RUN')
    parser.add_argument('--pattern', type=str,help='Input Type: short_sequence,long_segment')
    ###短序列可以通过文件输入、手动输入，如果不指明文件或手动输入则使用./fatsa_to_identify文件夹下所有文件
    ###长片段只有一种模式，读取./long_segment_to_identify文件夹下所有文件，同时可以使用阈值
    parser.add_argument('--model',default = "6.pth", type=str,  # "module.pth"
                        help='model name, The model needs to be in the folder ./save')         #导入模型
    parser.add_argument('--file_path',default ='none',type=str,
                        help='If no parameter is provided, all files in the path ./identify/fatsa_to_identify are used.  If it is not in the run folder, you must specify the path')
    parser.add_argument('--manual_input',default ='none',type=str,
                        help='If you are using manual input pattern, you need to enter the sequence with this option.')
    
    parser.add_argument('--limit',default = "0.9",type=str,
                        help='parameter:strict，normal，relax. relax pattern can identify more but with less precision but strict pattern opposite.')
    


    args = parser.parse_args()

    '''
    model_name = './save/'+args.model
    if torch.cuda.is_available()==False:
        model=torch.load(model_name,map_location='cpu')
        print('using cpu！')
    print('Model deployment completed')
    if torch.cuda.is_available():
        print('using cuda！')
        model = torch.load(model_name,map_location='cuda')
    model.eval()
    print(model)
    '''

    model_name = './save/'+args.model
    #model = ResNetAttention()
    model = ResNetSelfAttention()
    if torch.cuda.is_available()==False:
        model.load_state_dict(torch.load(model_name,map_location='cpu'))
        print('using cpu！')
    print('Model deployment completed')
    if torch.cuda.is_available():
        print('using cuda！')
        model = model.cuda()
        model.load_state_dict(torch.load(model_name,map_location='cuda'))
    model.eval()
    #print(model)


###If no files specified, all files will be used
    if(args.pattern=='short_sequence'):
        if(args.file_path == 'none'):
            print('use all fatsa_files!')
            all_DNA_path = glob.glob(r'./identify/fatsa_to_identify/*.fa')
            file1 = open("./identify/result_out.txt",'w')
            for DNA in all_DNA_path:
                result_,prob_ = fasta_base(DNA,model)
                #写入表格中
                write_line =DNA+'   '+str(result_)+'    '+str(prob_)
                file1.writelines(write_line+'\n')
                print(DNA)
            file1.close()

        if(args.file_path != 'none'):
###manual_input
            if(args.manual_input!='none'):
                seq = args.manual_input
                print('Use manual input!')
                result_,prob_ = manual_base(seq,model)
                print('The probability that input is ecc: ',prob_)
###Specified file
            else:
                print('use the fasta file:',args.file_path)
                result_,prob_ = fasta_base(args.file_path,model)
                print('The probability that ',args.file_path,' is ecc: ',prob_)

###对于长片段的识别
    '''
    1.使用两层滑动窗口，400bp窗口、10bp步长滑动后，得到(seq_lenth-400)/10的可能性序列
    2.设置阈值，高于n(0-100%)
    3.从滑动结果入手，再次滑动窗口，以10bp滑动
        找到第一个大于n-0.1的窗口，滑动5bp如果仍然大于阈值就计数
        直到连续两个窗口小于阈值

    细节：
    1.只有基于文件，一个fa文件创建一个表格来输出结果
    2.读取文件，先向量化再切片，输入(4,n)的矩阵，使用seq[:,x:x+400]切片
    3.复用dataprocess()进行向量化，复用forecast()函数进行400bp识别
    '''

    if(args.pattern=='long_segment'):
        print('long segment pattern!')
        print('use all fatsa_files in ./long_segment_to_identify!')
        if(args.file_path == 'none'):
            print('use all fatsa_files!')
            all_DNA_path = glob.glob(r'./identify/long_segment_to_identify/*.fa')
        else:
            all_DNA_path = glob.glob(str(args.file_path)+'/*.fa')
        for DNA in all_DNA_path:
            left,right,quantity,seq_clear = long_segment(DNA,model,args.limit)
            if quantity == 0:
                continue
            file2 = open("{}.txt".format(DNA),'w')
            debug = '-1'
            #写入表格中
            write_line1 = DNA +'   '+'eccDNA数量：'+str(quantity+1)
            file2.writelines(write_line1+'\n')
            print(write_line1)
            for wsoe in range(quantity+1):
                if(right[wsoe] != -1 and str(left[wsoe]) != debug):
                    write_line2 = '位置：'+'    '+str(left[wsoe]*30)+'  '+str(right[wsoe]*30)
                    debug = str(left[wsoe])
                    file2.writelines(write_line2+'\n')

                    fileseq = open("{}_{}.txt".format(DNA,str(wsoe+1)),'w')
                    write_line_seq = str(seq_clear[left[wsoe]*30:right[wsoe]*30])
                    fileseq.writelines(write_line_seq+'\n')
                    fileseq.close()

            print(DNA)
            file2.close()
        
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)
