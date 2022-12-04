from torch import cat
from torch.nn.modules.dropout import Dropout
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder11 = nn.Conv2d(n_channels, 64, 3, 1, 2, bias=False)
        self.encoder12 = nn.BatchNorm2d(64)
        self.encoder13 = nn.ReLU(inplace=True)

        self.encoder21 = nn.MaxPool2d(2)
        self.encoder22 = nn.Conv2d(64, 128, 3, 1, bias=False)
        self.encoder23 = nn.BatchNorm2d(128)
        self.encoder24 = nn.ReLU(inplace=True)

        self.encoder31 = nn.MaxPool2d(2)
        self.encoder32 = nn.Conv2d(128, 256, 3, 1, bias=False)
        self.encoder33 = nn.BatchNorm2d(256)
        self.encoder34 = nn.ReLU(inplace=True)

        self.encoder41 = nn.MaxPool2d(2)
        self.encoder42 = nn.Conv2d(256, 512, 3, 1, bias=False)
        self.encoder43 = nn.BatchNorm2d(512)
        self.encoder44 = nn.ReLU(inplace=True)
        
        self.encoder51 = nn.MaxPool2d(2)
        self.encoder52 = nn.Conv2d(512, 1024, 3, 1, bias=False)
        self.encoder53 = nn.BatchNorm2d(1024)
        self.encoder54 = nn.ReLU(inplace=True)

        # Decoder    
        self.decoder11 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.decoder12 = nn.Conv2d(1024, 512, 3, 1, bias=False)
        self.decoder13 = nn.BatchNorm2d(512)
        self.decoder14 = nn.ReLU(inplace=True)

        self.decoder21 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder22 = nn.Conv2d(512, 256, 3, 1, bias=False)
        self.decoder23 = nn.BatchNorm2d(256)
        self.decoder24 = nn.ReLU(inplace=True)

        self.decoder31 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder32 = nn.Conv2d(256, 128, 3, 1, bias=False)
        self.decoder33 = nn.BatchNorm2d(128)
        self.decoder34 = nn.ReLU(inplace=True)

        self.decoder41 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder42 = nn.Conv2d(128, 64, 3, 1, bias=False)
        self.decoder43 = nn.BatchNorm2d(64)
        self.decoder44 = nn.ReLU(inplace=True)

        self.decoder51 = nn.Conv2d(64, n_classes, 1)

    def padder(self, x1, x2_size):
        diffY = x2_size[2] - x1.size()[2]
        diffX = x2_size[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        
        x1 = cat([x1, x1], dim=1)

        return x1
        
    def forward(self, x):
        outputs = {}
        
        # Encoder
        #print(x.size())
        x = self.encoder11(x)
        x = self.encoder12(x)
        x = self.encoder13(x)
        x1_size = x.size()
        #print(x1_size)
        x = self.encoder21(x)
        x = self.encoder22(x)
        x = self.encoder23(x)
        x = self.encoder24(x)
        x2_size = x.size()
        #print(x2_size)
        x = self.encoder31(x)
        x = self.encoder32(x)
        x = self.encoder33(x)
        x = self.encoder34(x)
        x3_size = x.size()
        #print(x3_size)
        x = self.encoder41(x)
        x = self.encoder42(x)
        x = self.encoder43(x)
        x = self.encoder44(x)
        x4_size = x.size()
        #print(x4_size)
        x = self.encoder51(x)
        x = self.encoder52(x)
        x = self.encoder53(x)
        x = self.encoder54(x)
        # decoder
        x_decoder = self.decoder11(x)
        x_hat = self.padder(x_decoder, x4_size)
        x_hat = self.decoder12(x_hat)
        x_hat = self.decoder13(x_hat)
        x_hat = self.decoder14(x_hat)
        #print(x_hat.size())
        x_hat = self.decoder21(x_hat)
        x_hat = self.padder(x_hat, x3_size)
        x_hat = self.decoder22(x_hat)
        x_hat = self.decoder23(x_hat)
        x_hat = self.decoder24(x_hat)
        #print(x_hat.size())
        x_hat = self.decoder31(x_hat)
        x_hat = self.padder(x_hat, x2_size)
        x_hat = self.decoder32(x_hat)
        x_hat = self.decoder33(x_hat)
        x_hat = self.decoder34(x_hat)
        #print(x_hat.size())
        x_hat = self.decoder41(x_hat)
        x_hat = self.padder(x_hat, x1_size)
        x_hat = self.decoder42(x_hat)
        x_hat = self.decoder43(x_hat)
        x_hat = self.decoder44(x_hat)
        #print(x_hat.size())
        x_hat = self.decoder51(x_hat)
        #print('End decoder', x_hat.size())
        
        return {
            'z': x,
            'x_hat': x_hat
        }