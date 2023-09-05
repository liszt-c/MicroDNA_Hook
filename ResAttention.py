import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

LAYER_SIZE = 8
NUM_CLASS = 2
POLING_KERNEL = 5
LENGTH = 400
SCALING = 4

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv1d(in_channels, in_channels // SCALING, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // SCALING, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)
        out = self.gamma * out + x

        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetSelfAttention(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2, 2, 2], num_classes=NUM_CLASS):
        super(ResNetSelfAttention, self).__init__()
        self.in_planes = LAYER_SIZE
        self.shared_layers = nn.Sequential(
        nn.Conv1d(4, LAYER_SIZE, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm1d(LAYER_SIZE),
        self._make_layer(block, LAYER_SIZE, num_blocks[0], stride=1),
        SelfAttention(LAYER_SIZE),
        self._make_layer(block, LAYER_SIZE * 2, num_blocks[1], stride=2),
        SelfAttention(LAYER_SIZE * 2),
        self._make_layer(block, LAYER_SIZE * 4, num_blocks[2], stride=2),
        SelfAttention(LAYER_SIZE * 4),
        nn.AvgPool1d(kernel_size=POLING_KERNEL)           #(batchsize, LAYER_SIZE * 4, length/4)
        )

        self.heads = nn.Sequential(
            nn.Linear(int(LAYER_SIZE * 4*(LENGTH/4/POLING_KERNEL)), int(LAYER_SIZE * 4*(LENGTH/4/POLING_KERNEL)/20)), 
            nn.ReLU(),
            nn.Linear(int(LAYER_SIZE * 4*(LENGTH/4/POLING_KERNEL)/20), int(LAYER_SIZE * 4*(LENGTH/4/POLING_KERNEL)/160)),
            nn.ReLU(),
            nn.Linear(int(LAYER_SIZE * 4*(LENGTH/4/POLING_KERNEL)/160), 2)
            )
        self.multihead_attn = nn.MultiheadAttention(embed_dim=LAYER_SIZE*4, num_heads=8)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.shared_layers(x)
        #print(out.shape)
        out = out.view(out.size(0), -1) 
        out = self.heads(out)
        return out
    
if __name__ == '__main__':
    
    x = torch.randn(size=(64,4,400))          ###for test #(批处理维度,通道维度,长度)
    model = ResNetSelfAttention()

    output = model(x)
    print(output)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    '''
    from torch.profiler import profile, record_function, ProfilerActivity
    model=ResNetSelfAttention().cuda()
    inputs=torch.randn(size=(64,4,1000)).cuda()

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))##根据耗时排序
    '''



'''     out = self.conv1(x)             
        out = self.bn1(out)             
        out = F.relu(out)               
        out = self.layer1(out)          #(batchsize, 128, 1000)
        out = self.attention1(out)      #(batchsize, 128, 1000)
        out = self.layer2(out)          #(batchsize, 256, 500)
        out = self.attention2(out)      #(batchsize, 256, 500)
        out = self.layer3(out)          #(batchsize, 512, 250)
        out = self.attention3(out)      #(batchsize, 512, 250)
        out = self.avgpool(out)         #(batchsize, 512, 25)
'''