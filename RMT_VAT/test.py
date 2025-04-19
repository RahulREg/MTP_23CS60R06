# import os
# from PIL import Image
# import numpy as np

# folder_path = "../../../sugarbeet/gt/"
# all_unique_values = set()

# # List all PNG images
# image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# for img_file in sorted(image_files):
#     img_path = os.path.join(folder_path, img_file)
#     try:
#         # target = Image.open(img_path).convert("I")  # 32-bit grayscale
        
#         target = Image.open(img_path).convert("L")
        
#         target = target.resize((256, 256), Image.NEAREST)
        
#         # target = np.array(target).astype(np.float32) / 255.0

#         # target = np.array(target, dtype=np.uint8) 
#         unique_vals = np.unique(target)
#         print(f"{img_file}: {unique_vals}")
#         all_unique_values.update(unique_vals)
#     except Exception as e:
#         print(f"Failed to load {img_file}: {e}")

# # Print all unique values across all images
# print("Total unique values across all images:")
# print(sorted(all_unique_values))


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import numpy as np


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict)
        print("successfully load pretrained")
    return model


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:30:30 2019

@author: mwa
"""

from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms

class UnetBlock(nn.Module):
    def __init__(self, up_in1, up_out):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(up_out)

        # self.deconv = nn.ConvTranspose2d(size, size, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)


        #  init my layers
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


    def forward(self, up_p, x_p):

        # up_p = F.upsample(up_p, scale_factor=2, mode='bilinear', align_corners=True)

        # up_p = self.deconv(up_p)
        
        # up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = F.interpolate(up_p, size=x_p.shape[2:], mode='bilinear', align_corners=True)
        
        # cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = torch.add(up_p, x_p)


        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p   

class UnetBlock3d(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv3d(up_in1+up_in2, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm3d(up_out)


    def forward(self, up_p, x_p):

        n,c,rows,cols,deps = x_p.shape
        
        up_p = F.upsample(up_p, size=(rows,cols,deps), mode='trilinear')
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p   

class UnetBlock_(nn.Module):
    def __init__(self, up_in1, up_in2, up_out):
        super().__init__()

        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(up_out)


        # self.deconv = nn.ConvTranspose2d(2208, 2208, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)

        #  init my layers
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def center_crop(tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        return transforms.CenterCrop([h, w])(tensor)

    # def forward(self, up_p, x_p):

    #     up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
    #     # up_p = self.deconv(up_p)
    #     x_p = self.x_conv_(x_p)
    #     print(up_p.shape, x_p.shape)
    #     cat_p = torch.add(up_p, x_p)
    #     cat_p = self.x_conv(cat_p)
    #     cat_p = F.relu(self.bn(cat_p))

    #     return cat_p
    
    def forward(self, up_p, x_p):
        # Upsample up_p
        up_p = F.interpolate(up_p, size=x_p.shape[2:], mode='bilinear', align_corners=True)

        # Convolve x_p
        x_p = self.x_conv_(x_p)

        # Debug print shapes
        # print(f"up_p shape: {up_p.shape}, x_p shape: {x_p.shape}")

        # Add the two tensors
        cat_p = torch.add(up_p, x_p)

        # Further processing
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class UnetBlock3d_(nn.Module):
    def __init__(self, up_in1,up_in2,up_out):
        super().__init__()

        self.x_conv = nn.Conv3d(up_in1*2, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv3d(up_in2, up_in1, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm3d(up_out)


    def forward(self, up_p, x_p):

        n,c,rows,cols,deps = x_p.shape
        
        up_p = F.upsample(up_p, size=(rows,cols,deps), mode='trilinear')
        x_p = self.x_conv_(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))
                
        return cat_p 

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:29:18 2019

@author: mwa
"""

from torch import nn
# from models.densenet3d import densenet59
# from models.densenet import densenet121, densenet169, densenet201, densenet161
import torch.nn.functional as F
# from models.layers import SaveFeatures, UnetBlock_, UnetBlock, UnetBlock3d_, UnetBlock3d
import torch

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
    
class DenseUnet_2d(nn.Module):

    def __init__(self, densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        # self.up1 = UnetBlock_(2208,2112,768)
        # self.up2 = UnetBlock(768,384,768)
        # self.up3 = UnetBlock(384,96, 384)
        # self.up4 = UnetBlock(96,96, 96)

        self.up1 = UnetBlock_(2208, 2112, 768)
        self.up2 = UnetBlock(768, 384)
        self.up3 = UnetBlock(384, 96)
        self.up4 = UnetBlock(96, 96)


        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1, padding=0)

        # self.deconv = nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        # nn.init.xavier_normal_(self.deconv.weight)


        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)


        # x_fea = self.deconv(x)
        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_fea = self.conv1(x_fea)
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)
        x_fea = F.relu(self.bn1(x_fea))
        x_out = self.conv2(x_fea)

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove()
        
class hybridnet(nn.Module):

    def __init__(self, densenet='densenet59'):
        super().__init__()
        
        self.denseunet_2d = DenseUnet_2d()
        model_path = './models_save/DenseUnet_2d/30.pkl'
        model_dict = torch.load(model_path)
        model_dict_clone = model_dict.copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        self.denseunet_2d.load_state_dict(model_dict,False)
       
        base_model = densenet59
        layers = list(base_model().children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        self.up1 = UnetBlock3d_(504,496,504)
        self.up2 = UnetBlock3d(504,224,224)
        self.up3 = UnetBlock3d(224,192,192)
        self.up4 = UnetBlock3d(192,96,96)

        self.bn1 = nn.BatchNorm3d(64) 
        self.bn2 = nn.BatchNorm3d(64) 
        self.conv1 = nn.Conv3d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 3, kernel_size=1, padding=0)

    def forward(self, input):       
        n,c,h,w,d = input.shape
        x_volumes = x2d_to_volumes(input)
        out_2d,fea_2d = self.denseunet_2d(x_volumes)
        out_3d,fea_3d = dim_tran(out_2d)*250,dim_tran(fea_2d)
        x_3d = torch.cat((input,out_3d),1)
        
        x_3d = F.relu(self.rn(x_3d))
        x_3d = self.up1(x_3d, self.sfs[3].features)
        x_3d = self.up2(x_3d, self.sfs[2].features)
        x_3d = self.up3(x_3d, self.sfs[1].features)
        x_3d = self.up4(x_3d, self.sfs[0].features)

        x_out = F.upsample(x_3d, size=(h,w,d), mode='trilinear')
        x_out = self.conv1(x_out)
        x_out = F.dropout3d(x_out,p=0.3)
        x_out = F.relu(self.bn1(x_out))
        
        x_out = x_out + fea_3d
        x_out = self.conv2(x_out)
        x_out_dropout = F.dropout3d(x_out,p=0.1)
        x_out_bn = F.relu(self.bn2(x_out_dropout))
        final_result = self.conv3(x_out_bn)
        
        return final_result

    def close(self):
        for sf in self.sfs: sf.remove()  
        
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        out = nn.Softmax(dim=1)(c10)

        return out

        
        
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt

# import models.segnet as models
import models.erfnet as models

from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np

def compute_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - intersection
    iou = intersection / np.maximum(union, 1)
    return iou

def evaluate_model(model, dataloader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            outputs = outputs['out'] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    iou = compute_iou(cm)

    return acc, prec, rec, iou.mean(), iou


# Function to load the student and teacher models
def load_model(model_path):
    # student_model = DenseUnet_2d().cuda()
    # teacher_model = DenseUnet_2d().cuda()
    
    student_model = Unet(3,3).cuda()
    teacher_model = Unet(3,3).cuda()

    # student_model = models.SegNet(3,3).cuda()
    # teacher_model = models.SegNet(3,3).cuda()

    # student_model = models.ERFNet(3).cuda()
    # teacher_model = models.ERFNet(3).cuda()
    
    # weights = DeepLabV3_ResNet101_Weights.DEFAULT
    # student_model = deeplabv3_resnet101(weights=weights)
    # teacher_model = deeplabv3_resnet101(weights=weights)
    # # Modify classifier head
    # num_classes = 3  # soil, crop, weed
    # student_model.classifier = DeepLabHead(2048, num_classes)
    # teacher_model.classifier = DeepLabHead(2048, num_classes)
    # student_model = student_model.cuda()
    # teacher_model = teacher_model.cuda()
    
    # 2. Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    # 3. Load the model weights
    student_model.load_state_dict(checkpoint['state_dict'])
    teacher_model.load_state_dict(checkpoint['ema_state_dict'])
    
    # student_state_dict = torch.load('output/skin/skin50_tcsm/Mean-Teacher/unet1/student_sun_49.pth', weights_only=True)
    # teacher_state_dict = torch.load('output/skin/skin50_tcsm/Mean-Teacher/unet1/teacher_sun_49.pth', weights_only=True)
    
    # # Load weights into models
    # student_model.load_state_dict(student_state_dict)
    # teacher_model.load_state_dict(teacher_state_dict)
    
    # Set models to evaluation mode
    student_model.eval()
    teacher_model.eval()
    
    print(f"Loaded models from {student_path}")
        
    return student_model, teacher_model

# Function to predict a mask for a given image
def predict_image(model, image_path, transform):
    # Load and preprocess the input image
    # img = Image.open(image_path)
    # img = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension
    
    # import torch
    # from PIL import Image
    # import numpy as np

    img = Image.open(image_path).convert("RGB").resize((256, 256))  # resize manually
    img = np.array(img, dtype=np.float32) / 255.0  # manually divide by 255
    img = torch.from_numpy(img).permute(2, 0, 1)  # convert to [C, H, W]
    img = img.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')


    
    # Predict the mask
    with torch.no_grad():
        output = model(img)
        output = output['out'] if isinstance(output, dict) else output
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Convert to class predictions
    return predicted_mask

if __name__ == '__main__':
    # Paths to saved models
    # student_path = 'output/skin/skin50_tcsm/rmt_vat/rmt_unet_best.pth.tar'
    # student_path = '../output/skin/skin50_tcsm/rmt_vat/rmt_vat_model_best_50.pth.tar'
    # teacher_path = '/kaggle/working/final_teacher_model.pth'
    # student_path = '/kaggle/input/test_mean_teacher/pytorch/default/1/final_student_model (2).pth'
    # teacher_path = '/kaggle/input/test_mean_teacher/pytorch/default/1/final_teacher_model (2).pth'
    
    # Paths to dataset
    # rgb_folder = "../../../Data/test/rgb"
    # gt_folder = "../../../Data/test/gt"
    
    rgb_folder = "../../sunflower/val/rgb"
    gt_folder = "../../sunflower/val/gt"
    
    # rgb_folder = "../../val/rgb"
    # gt_folder = "../../val/gt"
    
    # Load models
    student_path = 'output/skin/skin50_tcsm/rmt_vat/rmt_unet_35_best.pth.tar'
    student_model, teacher_model = load_model(student_path)
    
    # Define the image transformation
    from torchvision.transforms import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Loop through all images in the folder
    import os

    # Define output folder for saving plots
    output_folder = "visualization/sun_rmt_unet_35"
    # output_folder = "visualization/sugar"
    
    os.makedirs(output_folder, exist_ok=True)  # Create it if it doesn't exist

    for image_name in os.listdir(rgb_folder):
        # Skip non-image files
        if not image_name.endswith('.png'):
            continue

        # Paths for input image and ground truth mask
        image_path = os.path.join(rgb_folder, image_name)
        gt_path = os.path.join(gt_folder, image_name)

        # Predict the mask
        prediction = predict_image(student_model, image_path, transform)
        # prediction = predict_image(teacher_model, image_path, transform)
        print("Unique predicted values:", np.unique(prediction))

        # Load the ground truth mask
        gt_mask = Image.open(gt_path).convert("L").resize((256, 256))
        gt_mask = np.array(gt_mask, dtype=np.float32)

        # Display the image, ground truth, and prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Image.open(image_path))
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid memory buildu

    # Define output folder for saving plots
    output_folder = "visualization/sun_rmt_unet_35_ema"
    # output_folder = "visualization/sugar"
    
    os.makedirs(output_folder, exist_ok=True)  # Create it if it doesn't exist

    for image_name in os.listdir(rgb_folder):
        # Skip non-image files
        if not image_name.endswith('.png'):
            continue

        # Paths for input image and ground truth mask
        image_path = os.path.join(rgb_folder, image_name)
        gt_path = os.path.join(gt_folder, image_name)

        # Predict the mask
        # prediction = predict_image(student_model, image_path, transform)
        prediction = predict_image(teacher_model, image_path, transform)
        print("Unique predicted values:", np.unique(prediction))

        # Load the ground truth mask
        gt_mask = Image.open(gt_path).convert("L").resize((256, 256))
        gt_mask = np.array(gt_mask, dtype=np.float32)

        # Display the image, ground truth, and prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Image.open(image_path))
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid memory build

    import utils.semantic_seg as transform
    import dataset.skinlesion as dataset
    import torch.utils.data as data
    
    transform_train = transform.Compose([
        transform.RandomRotationScale(),
        transform.ToTensor(),
        # transform.Normalize(mean=mean, std=std)
    ])

    transform_val = transform.Compose([
        transform.ToTensor(),
        # transform.Normalize(mean=mean, std=std)
    ])
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("../../sunflower/",
                                                                                                transform_train=transform_train,
                                                                                                transform_val=transform_val)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False)
    
    # assuming val_loader is your validation dataloader
    student_metrics = evaluate_model(student_model, val_loader, device)
    teacher_metrics = evaluate_model(teacher_model, val_loader, device)

    # Unpack
    s_acc, s_prec, s_rec, s_miou, s_iou_per_class = student_metrics
    t_acc, t_prec, t_rec, t_miou, t_iou_per_class = teacher_metrics

    # Print
    print("📘 Student Model:")
    print(f"Accuracy: {s_acc:.4f}")
    print(f"Precision: {s_prec:.4f}")
    print(f"Recall: {s_rec:.4f}")
    print(f"Mean IoU: {s_miou:.4f}")
    print(f"Per-class IoU: Soil: {s_iou_per_class[0]:.4f}, Crop: {s_iou_per_class[1]:.4f}, Weed: {s_iou_per_class[2]:.4f}\n")

    print("📗 Teacher Model:")
    print(f"Accuracy: {t_acc:.4f}")
    print(f"Precision: {t_prec:.4f}")
    print(f"Recall: {t_rec:.4f}")
    print(f"Mean IoU: {t_miou:.4f}")
    print(f"Per-class IoU: Soil: {t_iou_per_class[0]:.4f}, Crop: {t_iou_per_class[1]:.4f}, Weed: {t_iou_per_class[2]:.4f}")