import sys

sys.path.append("PerceptualSimilarity\\")
import os
import torch
import numpy as np
from torch import nn
from kornia import color
import torch.nn.functional as F
import cv2
from einops import rearrange, repeat
from torchvision import transforms
from torchvision.utils import save_image

class FC(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            elif self.activation == 'tanh':
                outputs = nn.Tanh()(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.conv1 = Conv2D(4, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(68, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation='tanh')

    def forward(self, inputs):
        secret, image = inputs
        # secret = secret - .5
        image = image - .5

        inputs = torch.cat([secret, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)
        # up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        # merge6 = torch.cat([conv4, up6], dim=1)
        # conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv4))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = 0.02 * self.residual(conv9)

        return residual


# class Extractor(nn.Module):
#     def __init__(self, secret_size=30):
#         super(Extractor, self).__init__()
#         self.secret_size = secret_size
#         self.extractor = nn.Sequential(
#             Conv2D(3, 32, 3, strides=2, activation='relu'),
#             Conv2D(32, 32, 3, activation='relu'),
#             Conv2D(32, 64, 3, strides=2, activation='relu'),
#             Conv2D(64, 64, 3, activation='relu'),
#             Conv2D(64, 64, 3, strides=2, activation='relu'),
#             Conv2D(64, 128, 3, strides=2, activation='relu'),
#             Conv2D(128, 128, 3, strides=2, activation='relu'),
#             Flatten(),
#             nn.Linear(6272, 40000),
#             nn.LayerNorm(40000),
#         )
            
#     def forward(self, image):
#         image = image - .5
#         image = self.extractor(image)
#         image = rearrange(image, 'b (1 h w) -> b 1 h w', h=200)
#         return image

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = Conv2D(3, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(67, 32, 3, activation='relu')
        self.output = Conv2D(32, 1, 1, activation=None)

    def forward(self, inputs):
        image = inputs
        image = image - .5
        
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # conv5 = self.conv5(conv4)
        # up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        # merge6 = torch.cat([conv4, up6], dim=1)
        # conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv4))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        output = self.output(conv9)
        return output
