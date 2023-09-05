#需求
#1.读取模型和参数
#2.调用dataloader读取数据、标签
#3.读出来的数据进模型跑，像tset中的一样


'''
1.dataprocess、dataloader原样，文件夹中只有eccDNA
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
    ROC_SMOOTH = 50                ###ROC曲线的光滑度
    ROC_row = [0]*ROC_SMOOTH            ###使用列表保存roc曲线的坐标
    ROC_col = [0]*ROC_SMOOTH
    ROC_total_num_labels_ecc = [0]*ROC_SMOOTH
    ROC_total_num_labels_notecc = [0]*ROC_SMOOTH
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
            #ROC          
            ### TP  FN  P
            ### FP  TN  N

            outputs_norm = torch.nn.functional.softmax(outputs,dim = -1)       ###在行维度进行softmax
            ROC_w,ROC_z = torch.split(outputs_norm,1,dim=1)         ###拆分出第一列存于ROC_w中
            for ROC_I in range(ROC_SMOOTH):                                ###设置滑块
                ROC_slide = ROC_I/ROC_SMOOTH                    ###注意ROC_I的第一项是0
                ROC_P_num = 0
                ROC_N_num = 0
                ROC_cout = 0
                ROC_acc_ecc = 0
                ROC_fcc_ecc = 0
                for ROC_t in test_labels.argmax(1):
                    ROC_cout = ROC_cout+1
                    if(ROC_t.item() == 0):                              ###标签是eccDNA
                        ROC_P_num = ROC_P_num+1                             ###标签是eccDNA的数量，P值
                        if(ROC_w[ROC_cout-1].item() >= ROC_slide):      ###大于等于该阈值则视为eccDNA
                            ROC_acc_ecc = ROC_acc_ecc +1                ###视为eccDNA的数量，TP值
                    if(ROC_t.item() == 1):                              ###标签是其他DNA
                        ROC_N_num = ROC_N_num +1
                        if(ROC_w[ROC_cout-1].item() >= ROC_slide):      ###其他DNA被视为eccDNA,FP值
                            ROC_fcc_ecc = ROC_fcc_ecc +1

                ROC_row[ROC_I] = ROC_row[ROC_I] + ROC_acc_ecc           ###TP
                ROC_total_num_labels_ecc[ROC_I] = ROC_total_num_labels_ecc[ROC_I] + ROC_P_num           ###P
                ROC_col[ROC_I] = ROC_col[ROC_I] + ROC_fcc_ecc           ###FP
                ROC_total_num_labels_notecc[ROC_I] = ROC_total_num_labels_notecc[ROC_I] + ROC_N_num     ###N


        line1 = "test set accuracy: {}".format((total_accuracy_ecc+total_accuracy_o)/(total_num_labels_ecc+total_num_labels_o))
        line2 = "test accuracy of eccDNA: {}".format(total_accuracy_ecc/total_num_labels_ecc)
        print(line1)
        print(line2)
        file2 = open('./acc_v'+str(version)+'.txt','w')
        file2.writelines(line1+'\n')
        file2.writelines(line2+'\n')
        file2.close()


        ###绘制ROC
        print('draw ROC')
        #for xxxx in ROC_row:print(xxxx)          ###for test
        file1 = open('./ROC/ROC_result_v'+str(version)+'.txt','w')
        for ROC_j in range(ROC_SMOOTH):
            #print(ROC_row[ROC_j],ROC_total_num_labels_ecc[ROC_j])       ###for test
            ROC_true = ROC_row[ROC_SMOOTH-ROC_j-1]/ROC_total_num_labels_ecc[ROC_SMOOTH-ROC_j-1]     ###先高阈值再低阈值
            #print(ROC_true)                                             ###for test
            ROC_false = ROC_col[ROC_SMOOTH-ROC_j-1]/ROC_total_num_labels_notecc[ROC_SMOOTH-ROC_j-1]
            write_line=str(ROC_true)+'   '+str(ROC_false)
            file1.writelines(write_line+'\n')
        file1.close()