import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from transformer2 import TransformerClassifier
from dataloader_transformer import data_for_run
import numpy as np
from torch.optim import lr_scheduler
import argparse
import datetime
import torch.nn.functional as F

filex = open(".\\train_out.txt",'w')

BATCH_SIZE = 32
EPOCH = 40
LEARN_STEP = 0.001                          #学习率
FloodingDepth = 0.0001                       #泛洪法正则化参数
DECREASING_LEARN_STEP = True               #衰减学习率
DCREASING_STEP_SIZE = 4                    #衰减间隔步数
DCREASING_GAMMA =0.6                        #衰减率
L2_DECAY = 1e-3                             #L2正则化

NUNBER_CHANNELS = 4
NUMBER_CLASSES = 2

if __name__ == '__main__':
    writer = SummaryWriter()
    train_dataloader,test_dataloader = data_for_run(BATCH_SIZE)

    ###Invoke cuda device
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("cuda available")
    else:
        device = torch.device("cpu")
        print("not found cuda")


    parser = argparse.ArgumentParser(description = 'test')
    parser.add_argument('--o',default = "./save", type=str,help='output dir.')
    args = parser.parse_args()

    model = TransformerClassifier()
    train_data = train_dataloader
    loss_method = torch.nn.CrossEntropyLoss().to(device)
    #loss_method = torch.nn.BCELoss().to(device)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_method = loss_method.cuda()
    learnstep = LEARN_STEP
    optimizer = torch.optim.Adam(model.parameters(),lr=learnstep,weight_decay=L2_DECAY)
    if(DECREASING_LEARN_STEP == True):  #衰减学习率
        scheduler = lr_scheduler.StepLR(optimizer, step_size=DCREASING_STEP_SIZE, gamma=DCREASING_GAMMA)

    epoch = EPOCH
    train_step = 0
    for i in range(epoch):
        print("-------epoch {}".format(i+1))
        model.train()

        for step, [DNAs, labels] in enumerate(train_data): 

            DNAs = torch.LongTensor(DNAs)
            labels = torch.Tensor(labels).float()
            #labels = labels.repeat(NUNBER_CHANNELS, 1)
            #labels = labels.view(-1, NUMBER_CLASSES)

            if torch.cuda.is_available():
                DNAs=DNAs.cuda()
                labels=labels.cuda()
            outputs = model(DNAs)
            #print(outputs.shape,labels.shape)
            #print(outputs,labels)
            loss = loss_method(outputs,labels)#.float()
            #loss = loss.type(torch.float).to(device)

            #优化器部分
            optimizer.zero_grad()
            b = FloodingDepth
            flood = (loss-b).abs()+b                           
            flood.backward()
            optimizer.step()
            train_step = len(train_dataloader)*i+step+1
            if train_step % 100 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("train time:{}, Loss: {}".format(train_step, loss.item()),time)
                writer.add_scalar("train_loss", loss.item(), train_step)

        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        est_data_lengt = 0
        test_data = test_dataloader
        total_accuracy_ecc = 0
        total_num_labels_ecc = 0
        test_step = 0
        ROC_SMOOTH = 50                ###ROC曲线的光滑度
        ROC_row = [0]*ROC_SMOOTH            ###使用列表保存roc曲线的坐标
        ROC_col = [0]*ROC_SMOOTH
        ROC_total_num_labels_ecc = [0]*ROC_SMOOTH
        ROC_total_num_labels_notecc = [0]*ROC_SMOOTH
        with torch.no_grad():
            for test_data_length, [test_DNAs, test_labels] in enumerate(test_data):
                
                #test_labels = np.array(test_labels)  
                #test_labels = torch.Tensor(test_labels).long()                ###V3损失函数要求labels是64位.long()
                #test_labels = torch.Tensor(test_labels).float()      ###V5修改优化器并更改标签格式后需.float()
                test_DNAs = torch.LongTensor(test_DNAs)
                test_labels = torch.Tensor(test_labels).float()
                #test_labels = test_labels.repeat(NUNBER_CHANNELS, 1)
                #test_labels = test_labels.view(-1, NUMBER_CLASSES)
                if torch.cuda.is_available():
                    test_DNAs = test_DNAs.cuda()
                    test_labels = test_labels.cuda()
                outputs = model(test_DNAs)
                loss = loss_method(outputs, test_labels)
                total_test_loss = total_test_loss + loss.item()
                test_step = len(test_dataloader)*i+test_data_length+1
                if test_step % 25 == 0:
                    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("test time:{}, Loss: {}".format(test_step, loss.item()),time)



                write_line1 = 'outputs\n'+ str(outputs) + 'test_labels\n'+ str(test_labels) + '\n'
                filex.writelines(write_line1+'\n')



                accuracy = (outputs.argmax(1) == test_labels.argmax(1)).sum()               ###v5修改比较方式
                total_accuracy = total_accuracy + accuracy
                
                #eccDNA_accuracy
                cout = 0
                acc_ecc = 0
                num = 0
                for t in test_labels.argmax(1):
                    cout = cout + 1
                    t_ecc = t.item()
                    if(t_ecc == 0):     #标签是eccDNA,即标签为(1,0),其.argmax(1)为0
                        num = num+1
                        if(outputs.argmax(1)[cout-1].item() == 0):
                            acc_ecc = acc_ecc+1
                total_accuracy_ecc = total_accuracy_ecc + acc_ecc
                total_num_labels_ecc = total_num_labels_ecc  + num

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

                #print(outputs,test_labels)                                                  ###for test
                #print(outputs.argmax(1),test_labels.argmax(1))                              ###for test
                #print(accuracy)                                                             ###for test
                #print(total_accuracy)                                                       ###for test
                #print(test_data_length)                                                     ###for test
            print("test set Loss: {}".format(total_test_loss))
            print("test set accuracy: {}".format(total_accuracy/test_data_length/BATCH_SIZE))
            print("test accuracy of eccDNA: {}".format(total_accuracy_ecc/total_num_labels_ecc))
            writer.add_scalar("test_loss", total_test_loss, i)
            writer.add_scalar("test_accuracy", total_accuracy/test_data_length/BATCH_SIZE, i)
            writer.add_scalar("test accuracy of eccDNA", total_accuracy_ecc/total_num_labels_ecc, i)
            ###绘制ROC
            print('draw ROC')
            #for xxxx in ROC_row:print(xxxx)          ###for test
            file1 = open('./ROC/ROC_result_'+str(i+1)+'.txt','w')
            for ROC_j in range(ROC_SMOOTH):
                #print(ROC_row[ROC_j],ROC_total_num_labels_ecc[ROC_j])       ###for test
                ROC_true = ROC_row[ROC_SMOOTH-ROC_j-1]/ROC_total_num_labels_ecc[ROC_SMOOTH-ROC_j-1]     ###先高阈值再低阈值
                #print(ROC_true)                                             ###for test
                ROC_false = ROC_col[ROC_SMOOTH-ROC_j-1]/ROC_total_num_labels_notecc[ROC_SMOOTH-ROC_j-1]
                write_line=str(ROC_true)+'   '+str(ROC_false)
                file1.writelines(write_line+'\n')
            file1.close()
                

        if(DECREASING_LEARN_STEP == True):
            scheduler.step()    #衰减学习率计数
        
        torch.save(model, "{}/module_{}.pth".format(args.o,i+1))  ###注意文件夹
        torch.save(model.state_dict(),"{}/module_dict_{}.pth".format(args.o,i+1))
        print("saved epoch {}".format(i+1))
    filex.close()
    writer.close()