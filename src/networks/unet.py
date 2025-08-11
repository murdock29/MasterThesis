import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid


class UNet(nn.Module):
    """Modified version of https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3."""

    def __init__(self, channels=3):
        super().__init__()

        # main part of the network
        # input: 128x128x3
        self.e11 = nn.Conv2d(channels, 64, kernel_size=3, padding=0)  # output: 128x128x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=0)  # output: 128x128x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 62x62x64
        self.dropout1 = nn.Dropout()

        # input: 62x62x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 64x64x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 64x64x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x128
        self.dropout2 = nn.Dropout()

        # input: 29x29x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # output: 32x32x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 32x32x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 16x16x256
        self.dropout3 = nn.Dropout()

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 16x16x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 16x16x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 8x8x512
        self.dropout4 = nn.Dropout()

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 8x8x512
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 8x8x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=2)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=2)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=2)

        self.outconv = nn.Conv2d(64, out_channels=1, kernel_size=1)


    def forward(self, x):
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        xp1 = self.dropout1(xp1)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        xp2 = self.dropout2(xp2)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        xp3 = self.dropout3(xp3)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)
        xp4 = self.dropout4(xp4)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1) #xe42[:, :, 4:-4, 4:-4]
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1) #[:, :, 16:-16, 16:-16]
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1) #[:, :, 40:-40, 40:-40]
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1) #[:, :, 88:-88, 88:-88]
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        logits = self.outconv(xd42)
        return logits


def unet(pretrained=False, **kwargs):
    if pretrained:
        weights = {
            "v1": "https://github.com/JJGO/UniverSeg/releases/download/weights/universeg_v1_nf64_ss64_STA.pt"
        }
        model = UNet(**kwargs)
        state_dict = torch.hub.load_state_dict_from_url(weights["v1"])
        model.load_state_dict(state_dict)
    model = UNet(**kwargs)
    return model
