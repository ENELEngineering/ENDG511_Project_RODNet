import torch.nn as nn

# Parent class for multibranch model
class RODParent(nn.Module):
    def __init__(self):                     # constructor for initializing static variables
        super(RODParent, self).__init__()

        self.encoder_base = None
        self.decoder_long = None
        self.decoder_short = None
    
    def shortbranch_inf(self,X):            # method used during determining inference time of short branch
        X = self.encoder_base(X)
        X = self.decoder_short(X)
        return X
    
    def longbranch_inf(self,X):             # method used during determining inference time of long branch
        X = self.encoder_base(X)
        X = self.decoder_long(X)
        return X
    
    def compute_short_branch_size(self):        # method used to compute short branch model size
        param_size = 0
        for param in self.encoder_base.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.decoder_short.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.encoder_base.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.decoder_short.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def compute_long_branch_size(self):         # method used to compute long branch model size
        param_size = 0
        for param in self.encoder_base.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.decoder_long.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.encoder_base.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.decoder_long.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def print_summary(self):                # method used to display number of parameters 
        encoderbase_total_params = sum(p.numel() for p in self.encoder_base.parameters())
        decodershort_total_params = sum(p.numel() for p in self.decoder_short.parameters())
        decoderlong_total_params = sum(p.numel() for p in self.decoder_long.parameters())

        print("Number of encoder parameters: {}".format(encoderbase_total_params))
        print("Number of decoder short parameters: {}".format(decodershort_total_params))
        print("Number of decoder long parameters: {}".format(decoderlong_total_params))

# parent class for base model
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

# Base model class    
class ROD_V0(BaseParent):
    def __init__(self, in_channels, n_class):   # constructor for instantiating sequential pytorch neural network layers
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

# multibranch model class
class ROD_V1(RODParent):
    def __init__(self, in_channels, n_class):       # constructor for instantiating sequential pytorch neural network layers
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

# secondary multibranch model not used for training or validation
class ROD_V2(RODParent):
    def __init__(self, in_channels, n_class):
        super(ROD_V2, self).__init__()

        self.encoder_base = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2)),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64,
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