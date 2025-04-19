import torch
import torch.nn as nn

from models.layers import SaveFeatures, UnetBlock_, UnetBlock, UnetBlock3d_, UnetBlock3d
def ComputePara(net):
    params = list(net.parameters())
    k = 0
#    if not os.path.exists(savePath):
#         file_w = open(savePath,'w')
#    file_w = open(savePath,'r+')  
#    file_w.read()
    for i in params:
        l = 1
#        print("layer structure:" + str(list(i.size())))
#        file_w.write("layer structure:" + str(list(i.size())) + '\n') 
        for j in i.size():
            l *= j
#        print("layer paramenters:"+str(l))
#        file_w.write("layer paramenters:" + str(l) + '\n')
        k += l
    print("network paramenters:"+str(k))
#    file_w.write("network paramenters:" + str(k) + '\n') 
#    file_w.close()

def x2d_to_volumes(x):
    n,c,h,w,d = x.shape
    x_start = x[:,:,:,:,0:1]
    x_end = x[:,:,:,:,d-1:d]
    x = torch.cat((x_start,x,x_end),4)
    x_3d = x[:,0,:,:,0:3].permute(0,3,1,2)
    for i in range(1,d):
        x_tmp = x[:,0,:,:,i:i+3].permute(0,3,1,2)
        x_3d = torch.cat((x_3d,x_tmp),0)
    return x_3d

def dim_tran(x):      
    x = x.permute(1,2,3,0)
    x = x.unsqueeze(0)
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()

        self.layer4 = self._make_layer(512, 256, 2)
        self.layer3 = self._make_layer(256, 128, 2)
        self.layer2 = self._make_layer(128, 64, 2)
        self.layer1 = self._make_layer(64, 64, 2)
        
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.Sigmoid()  # or nn.Softmax(dim=1) if multi-class segmentation
        )

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.final_block(x)
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x, dropout = False):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

