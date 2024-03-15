import torch.nn as nn

class RODParent(nn.Module):
    def __init__(self):
        super(RODParent, self).__init__()

        self.encoder_base = None
        self.decoder_long = None
        self.decoder_short = None
    
    def print_summary(self):
        encoderbase_total_params = sum(p.numel() for p in self.encoder_base.parameters())
        decodershort_total_params = sum(p.numel() for p in self.decoder_short.parameters())
        decoderlong_total_params = sum(p.numel() for p in self.decoder_long.parameters())

        print("Number of encoder parameters: {}".format(encoderbase_total_params))
        print("Number of decoder short parameters: {}".format(decodershort_total_params))
        print("Number of decoder long parameters: {}".format(decoderlong_total_params))

class BaseParent(nn.Module):
    def __init__(self):
        super(BaseParent, self).__init__()

        self.encoder_base = None
        self.decoder_long = None
    
    def print_summary(self):
        encoderbase_total_params = sum(p.numel() for p in self.encoder_base.parameters())
        decoderlong_total_params = sum(p.numel() for p in self.decoder_long.parameters())

        print("Number of encoder parameters: {}".format(encoderbase_total_params))
        print("Number of decoder long parameters: {}".format(decoderlong_total_params)) 
    
class ROD_V0(BaseParent):
    def __init__(self, in_channels, n_class):
        super(ROD_V0, self).__init__()

        self.encoder_base = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=64,
                                        kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=64),
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=64,
                                        kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=64),
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=128,
                                        kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=128),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=128,
                                        kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=128),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=256,
                                        kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=256),
                nn.ReLU(),
                nn.Conv3d(in_channels=256, out_channels=256,
                                        kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=256),
                nn.ReLU(),
        )

        self.decoder_long = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.Sigmoid()
        )
        
        self.print_summary()

    def forward(self, x):
        x = self.encoder_base(x)
        x = self.decoder_long(x)
        return x

class ROD_V1(RODParent):
    def __init__(self, in_channels, n_class):
        super(ROD_V1, self).__init__()

        self.encoder_base = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=64,
                                        kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=64),
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=64,
                                        kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=64),
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=128,
                                        kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=128),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=128,
                                        kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2)),
                nn.BatchNorm3d(num_features=128),
                nn.ReLU(),
        )

        self.decoder_short = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.Sigmoid()
        )

        self.decoder_long= nn.Sequential(
            nn.MaxPool3d((2,1,1), stride = (2,1,1)),
            nn.Conv3d(in_channels=128, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2)),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                    kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.Sigmoid()
        )
        
        self.print_summary()

    def forward(self, x):
        x = self.encoder_base(x)
        x1 = self.decoder_short(x)
        x2 = self.decoder_long(x)
        return x1,x2