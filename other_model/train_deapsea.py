import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from ResNet_Attention import ResNetAttention
from dataloader_deepsea import data_for_run
import numpy as np
from torch.optim import lr_scheduler
import argparse
import datetime


BATCH_SIZE = 256
EPOCH = 40
LEARN_STEP = 0.001                          #学习率
FloodingDepth = 0.001                       #泛洪法正则化参数
DECREASING_LEARN_STEP = True               #衰减学习率
DCREASING_STEP_SIZE = 4                    #衰减间隔步数
DCREASING_GAMMA =0.6                        #衰减率，0.6衰减9次大概衰减为原来的0.01
L2_DECAY = 0 #1e-3                             #L2正则化

writer = SummaryWriter()
train_dataloader,test_dataloader = data_for_run(BATCH_SIZE)

###Invoke cuda device
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print("cuda available")
else:
    device = torch.device("cpu")
    print("not found cuda")

###实例化模型
model = ResNetAttention()
###training
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'test')
    parser.add_argument('--o',default = "./save", type=str,help='output dir.')
    args = parser.parse_args()

    model = ResNetAttention(in_channels=4,classes=919)
    train_data = train_dataloader
    loss_method = torch.nn.BCEWithLogitsLoss()      ###多分类多标签，sigmoid
    #loss_method = torch.nn.CrossEntropyLoss()      ###单/多分类单标签，softmax
    if torch.cuda.is_available():
        model = model.cuda()
        loss_method = loss_method.cuda()
    learnstep = LEARN_STEP

    optimizer = torch.optim.Adam(model.parameters(),lr=learnstep,weight_decay=L2_DECAY)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learnstep)
    if(DECREASING_LEARN_STEP == True):  #衰减学习率
        scheduler = lr_scheduler.StepLR(optimizer, step_size=DCREASING_STEP_SIZE, gamma=DCREASING_GAMMA)

    epoch = EPOCH
    train_step = 0
    for i in range(epoch):
        print("-------epoch {}".format(i+1))
        model.train()
        #print('learnstep=',learnstep)
        for step, [DNAs, labels] in enumerate(train_data): 
            #labels = torch.Tensor(labels).long()                ###V3损失函数要求labels是64位.long()
            labels = torch.Tensor(labels).float()               ###V5修改优化器并更改标签格式后需.float()
            if torch.cuda.is_available():
                DNAs=DNAs.cuda()
                labels=labels.cuda()
            outputs = model(DNAs)
            #print(outputs,labels)                               ###for test
            loss = loss_method(outputs,labels)

            #优化器部分
            optimizer.zero_grad()
            b = FloodingDepth
            flood = (loss-b).abs()+b                            ###V6.4更新flooding方法
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
        test_data = test_dataloader
        with torch.no_grad():
            for test_data_length, [test_DNAs, test_labels] in enumerate(test_data):
                #test_labels = np.array(test_labels)  
                #test_labels = torch.Tensor(test_labels).long()                ###V3损失函数要求labels是64位.long()
                test_labels = torch.Tensor(test_labels).float()      ###V5修改优化器并更改标签格式后需.float()
                if torch.cuda.is_available():
                    test_DNAs = test_DNAs.cuda()
                    test_labels = test_labels.cuda()
                outputs = model(test_DNAs)
                loss = loss_method(outputs, test_labels)
                total_test_loss = total_test_loss + loss.item()
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("test set Loss: {}".format(total_test_loss),time)                
            writer.add_scalar("test_loss", total_test_loss, i)

                
        if(DECREASING_LEARN_STEP == True):
            scheduler.step()    #衰减学习率计数
        
        torch.save(model, "{}/module_{}.pth".format(args.o,i+1))  ###注意文件夹
        torch.save(model.state_dict(),"{}/module_dict_{}.pth".format(args.o,i+1))
        print('\n')
        print("saved epoch {}".format(i+1))

