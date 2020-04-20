from torch import nn as nn
import torch


def conv44(in_size, out_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_size),
        nn.LeakyReLU(0.1, inplace=True)
    )

def Tranconv44(in_size, out_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_size),
        nn.LeakyReLU(0.1, inplace=True)
    )

class HEAR(nn.Module):
    def __init__(self):
        super(HEAR, self).__init__()
        self.layer1 = conv44(6,64)
        self.layer2 = conv44(64,128)
        self.layer3 = conv44(128,256)
        self.layer4 = conv44(256,512)
        self.layer5 = conv44(512,512)

        self.tranlayer1 = Tranconv44(512,512)
        self.tranlayer2 = Tranconv44(1024,256)
        self.tranlayer3 = Tranconv44(512,128)
        self.tranlayer4 = Tranconv44(256,64)
        self.tranlayer5 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        t1 = self.layer1(input)
        t2 = self.layer2(t1)
        t3 = self.layer3(t2)
        t4 = self.layer4(t3)
        t5 = self.layer5(t4)

        rt1 = self.tranlayer1(t5)
        rt1 = torch.cat((t4, rt1), dim=1)
        rt2 = self.tranlayer2(rt1)
        rt2 = torch.cat((t3, rt2), dim=1)
        rt3 = self.tranlayer3(rt2)
        rt3 = torch.cat((t2, rt3), dim=1)
        rt4 = self.tranlayer4(rt3)
        rt4 = torch.cat((t1, rt4), dim=1)

        upSample = nn.functional.interpolate(rt4, scale_factor=2, mode='bilinear', align_corners=True)
        rt5 = self.tranlayer5(upSample)
        # don't understand here
        result = torch.tanh(rt5)
        return result


        
