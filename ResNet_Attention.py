import torch
import torch.nn as nn


SIZEofMODULE = 64      ###模型中间宽度
CLASSES = 2


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.stride = stride
        self.out_channels = out_channels

        self.resconv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        self.resnorm = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5, affine=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or identity.size() != out.size():           #stride=!1 or in_chanel=!out_chanel，进行下采样，为卷积层
            identity = self.resconv(identity)
            identity = self.resnorm(identity)

        out += identity
        out = self.relu(out)

        return out

class AttentionBlock(nn.Module):                #空间注意力机制
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv_att = nn.Conv1d(in_channels, 1, kernel_size=1, stride=1, bias=False)
        self.softmax_att = nn.Softmax(dim=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        att = self.conv_att(x)                  #对通道卷积，多通道变为1通道
        att = self.softmax_att(att)             #在批次中做softmax，换成sigmod可能会更好，softmax效果可能好点，sigmod易学习
        out = x * att
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResNetAttention(nn.Module):
    def __init__(self,in_channels=4,classes=CLASSES):
        super(ResNetAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, SIZEofMODULE, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(SIZEofMODULE)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(SIZEofMODULE, SIZEofMODULE, 2, 1) #(64, 64, 2)
        self.layer2 = self._make_layer(SIZEofMODULE, SIZEofMODULE*2, 2, 2)#(64, 128, 2 ,stride=2)
        self.layer3 = self._make_layer(SIZEofMODULE*2, SIZEofMODULE*4, 2, 2)#(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(SIZEofMODULE*4, SIZEofMODULE*8, 2, 2)#(256, 512, 2, stride=2)
        self.attention = AttentionBlock(SIZEofMODULE*8, SIZEofMODULE*8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(SIZEofMODULE*8, classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=2):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):                                              #(1,2)，输出1
            layers.append(ResidualBlock(out_channels, out_channels,stride=1))            #in_chenel = out_chanel,stride = 1,不采样直接联通
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)           #展平层，多维度的tensor展平成一维
        x = self.fc(x)
        
        return x