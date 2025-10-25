import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import os
import time

def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        
class ProcessorLoss(nn.Module):
    def __init__(self):
        super(ProcessorLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs):
        mse_loss = self.mse(outputs, torch.zeros_like(outputs))
        return mse_loss
     
     
class ConvTransposeBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=4, stride=2, padding=1):
        super(ConvTransposeBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
    
    
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.apply(init_weights)
    def forward(self, x):
        return self.layers(x)
    
    
class MessageProcessor(nn.Module):
    def __init__(self):
        super(MessageProcessor, self).__init__()
        # 
        # self.enc1 = ConvBNRelu(1, 16)
        # self.enc2 = ConvBNRelu(16, 32)
        
        # self.bottleneck = ConvBNRelu(32, 64)
        
        # self.upconv2 = self.upconv(64, 32)
        # self.dec2 = ConvBNRelu(64, 32)
        # self.upconv1 = self.upconv(32, 16)
        # self.dec1 = ConvBNRelu(32, 16)
        
        # self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        # self.up = nn.Upsample(size=(200, 200))
        
        # Why so complex…… no need at all #
        # Complex model do harm to convergence, fuck... #

        self.linear_reflect = nn.Sequential(
            nn.Linear(900, 3000),
            nn.Linear(3000, 10000),
        )


    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(1)
        x = repeat(x, 'b c h w -> b c h (30 w)')
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.linear_reflect(x)

        # print(x.shape)
        # x = self.up(x)
        # # x  ->  b c 200 200 #
        # enc1 = self.enc1(x)  # Shape: (batch_size, 16, 200, 200)
        # enc2 = self.enc2(F.max_pool2d(enc1, 2))  # Shape: (batch_size, 32, 100, 100)
        
        # bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))
        
        # dec2 = self.upconv2(bottleneck)  # Shape: (batch_size, 32, 100, 100)
        # dec2 = torch.cat((dec2, enc2), dim=1)  # Shape: (batch_size, 64, 100, 100)
        # dec2 = self.dec2(dec2)  # Shape: (batch_size, 32, 100, 100)
        
        # dec1 = self.upconv1(dec2)  # Shape: (batch_size, 16, 200, 200)
        # dec1 = torch.cat((dec1, enc1), dim=1)  # Shape: (batch_size, 32, 200, 200)
        # dec1 = self.dec1(dec1)  # Shape: (batch_size, 16, 200, 200)
        
        # final = self.final_conv(dec1)
        x = rearrange(x, 'b c (h w) -> b c h w', h=100)
        up = nn.Upsample(scale_factor=(2, 2))
        x = up(x)
        return x
        # print(x.shape)
        # print(self.layers(x).shape)
        # return self.layers(x)
    
    
class MessageExtractor(nn.Module):
    def __init__(self):
        super(MessageExtractor, self).__init__()
        
        self.features = nn.Sequential(
            ConvBNRelu(1, 32),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBNRelu(128, 256),
            nn.MaxPool2d(5, 5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6400, 1024),
            nn.Linear(1024, 30),
        )
        
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.features(x)

        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.classifier(x)
        return torch.sigmoid(x)
