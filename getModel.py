# Copyright 2024. All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.
# 
# This python file is used explicitly to meet the project requirements provided
# in ENDG 511 at the University of Calgary.

import torch.nn as nn
import torch
import os


class RODEncodeBase(nn.Module):
    """
    This is the base RodNet encoder architecture.
    """

    def __init__(self, in_channels: int=2):
        super(RODEncodeBase, self).__init__()
        
        self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return x

class RODEncodeShort(nn.Module):
    """
    This is the shortened RodNet encoder architecture.
    """

    def __init__(self, in_channels: int=2):
        super(RODEncodeShort, self).__init__()
        
        self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        return x

class RODDecodeBase(nn.Module):
    """
    This is the base RodNet decoder architecture.
    """

    def __init__(self, n_class: int):
        super(RODDecodeBase, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        x = self.sigmoid(x)
        return x

class RODDecodeShort(nn.Module):
    """
    This is the shortened RodNet decoder branch architecture.
    """

    def __init__(self, n_class: int):
        super(RODDecodeShort, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.sigmoid(x)
        return x
    
class RODDecodeLong(nn.Module):
    """
    This is the elongated RodNet decoder branch architecture.
    """

    def __init__(self, n_class: int):
        super(RODDecodeLong, self).__init__()
        self.maxp = nn.MaxPool3d((2,1,1), stride = (2,1,1))
        self.conv1a = nn.Conv3d(in_channels=128, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=256, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                    kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.maxp(x)
        x = self.relu(self.bn(self.conv1a(x)))
        x = self.relu(self.bn(self.conv1b(x)))
        x = self.prelu(self.convt1(x))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        x = self.sigmoid(x)
        return x

class RODNetBase(nn.Module):
    """
    This architecture is the base RODNet model.
    """

    def __init__(self, in_channels: int, n_class: int):
        super(RODNetBase, self).__init__()
        self.encoder = RODEncodeBase(in_channels=in_channels)
        self.decoder = RODDecodeBase(n_class=n_class)
       
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def print_summary(self):
        encoder_total_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_total_params = sum(p.numel() for p in self.decoder.parameters())

        print("Number of encoder parameters: {}".format(encoder_total_params))
        print("Number of decoder long parameters: {}".format(decoder_total_params)) 

    def print_size_of_model(self):
        torch.save(self.state_dict(), "temp.p")
        print('Size (KB):', os.path.getsize("temp.p")/1e3)
        os.remove('temp.p')

    def print_parameter_type(self, cutoff: int=None):
        for i, (n, p) in enumerate(self.named_parameters()):
            print(n, ": ", p.dtype)
            if i == cutoff:
                break

class RODNetBranched(nn.Module):
    """
    This architecture is the branched RODNet model.
    """

    def __init__(self, in_channels: int, n_class: int):
        super(RODNetBranched, self).__init__()
        self.encoder = RODEncodeShort(in_channels=in_channels)
        self.decoder_short = RODDecodeShort(n_class=n_class)
        self.decoder_long = RODDecodeLong(n_class=n_class)
    
    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder_short(x)
        x2 = self.decoder_long(x)
        return x1, x2
    
    def print_summary(self):
        encoder_total_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_short_total_params = sum(p.numel() for p in self.decoder_short.parameters())
        decoder_long_total_params = sum(p.numel() for p in self.decoder_long.parameters())

        print("Number of encoder parameters: {}".format(encoder_total_params))
        print("Number of decoder short parameters: {}".format(decoder_short_total_params))
        print("Number of decoder long parameters: {}".format(decoder_long_total_params))

    def print_size_of_model(self):
        torch.save(self.state_dict(), "temp.p")
        print('Size (KB):', os.path.getsize("temp.p")/1e3)
        os.remove('temp.p')

    def print_parameter_type(self, cutoff: int=None):
        for i, (n, p) in enumerate(self.named_parameters()):
            print(n, ": ", p.dtype)
            if i == cutoff:
                break