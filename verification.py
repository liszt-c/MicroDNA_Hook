#
#1.读取模型和参数
#2.调用dataloader读取数据、标签
#3.读出来的数据进模型跑，像tset中的一样


'''
1.dataprocess、dataloader原样，文件夹中只有eccDNA，测试准确率
2.取训练模块中测试集的部分代码，作为验证
'''
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataloader_v13 import data_for_run
import numpy as np
from torch.optim import lr_scheduler
import argparse
import datetime

BATCH_SIZE = 3072
version = 10

###部署模型
from model import ResNet
model = ResNet()
#from ResNet_Attention_v126 import ResNetAttention
#model = ResNetAttention()
model_name = './save/v'+str(version)+'.pth'
if __name__=='__main__':

    if torch.cuda.is_available()==False:
        model.load_state_dict(torch.load(model_name,map_location='cpu'))
        print('using cpu！')
    print('Model deployment completed')
    if torch.cuda.is_available():
        print('using cuda！')
        model = model.cuda()
        model.load_state_dict(torch.load(model_name,map_location='cuda'))
    model.eval()


    ###读取数据
    test_data = data_for_run(BATCH_SIZE)


    total_accuracy_o = 0
    est_data_lengt = 0
    total_accuracy_ecc = 0
    total_num_labels_ecc = 0
    total_num_labels_o = 0
    all_legth = 0
    test_step = 0
    with torch.no_grad():
        for test_data_length, [test_DNAs, test_labels] in enumerate(test_data):
            print(test_data_length)
            test_labels = torch.Tensor(test_labels).float()      ###V5修改优化器并更改标签格式后需.float()
            if torch.cuda.is_available():
                test_DNAs = test_DNAs.cuda()
                test_labels = test_labels.cuda()
            outputs = model(test_DNAs)

            #eccDNA_accuracy
            cout = 0
            acc_ecc = 0
            acc_o = 0
            num = 0
            num_o = 0
            for t in test_labels.argmax(1):
                cout = cout + 1
                
                t_ecc = t.item()
                if(t_ecc == 0):     #标签是eccDNA,即标签为(1,0),其.argmax(1)为0
                    num = num+1
                    if(outputs.argmax(1)[cout-1].item() == 0):
                        acc_ecc = acc_ecc+1
                else:
                    num_o = num_o + 1 
                    if(outputs.argmax(1)[cout-1].item() == 1):
                        acc_o = acc_o+1
            all_legth = all_legth + cout
            total_accuracy_ecc = total_accuracy_ecc + acc_ecc
            total_accuracy_o = total_accuracy_o + acc_o

            total_num_labels_ecc = total_num_labels_ecc + num
            total_num_labels_o = total_num_labels_o +num_o


            if all_legth % 100 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("test time:{} ".format(all_legth),time)
    

        line1 = "test set accuracy: {}".format((total_accuracy_ecc+total_accuracy_o)/(total_num_labels_ecc+total_num_labels_o))
        line2 = "test accuracy of eccDNA: {}".format(total_accuracy_ecc/total_num_labels_ecc)
        print(line1)
        print(line2)
        file2 = open('./acc_v'+str(version)+'.txt','w')
        file2.writelines(line1+'\n')
        file2.writelines(line2+'\n')
        file2.close()
