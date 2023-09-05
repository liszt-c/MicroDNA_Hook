import torch

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.relu = torch.nn.ReLU(inplace=True)

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        
        x = self.layer(x)+residual
        x = self.relu(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self,in_channels=4,classes=919):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,512,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),
            
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),

            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),


            Bottlrneck(2048,1024,4096, True),
            Bottlrneck(4096,1024,4096, False),
            Bottlrneck(4096,1024,4096, False),
            Bottlrneck(4096,1024,4096, False),
            Bottlrneck(4096,1024,4096, False),
            Bottlrneck(4096,1024,4096, False),
                        
            Bottlrneck(4096,2048,8192,True),
            Bottlrneck(8192,2048,8192,False),
            Bottlrneck(8192,2048,8192,False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(8192,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,8192)
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    x = torch.randn(size=(64,4,1024))          ###for test #(批处理维度,通道维度,长度)
    model = ResNet(in_channels=4)

    output = model(x)
    print(output)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
