import torch
import torch.nn as nn
import torch.nn.functional as F
from AAD import *

def conv4x4(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )

def deconv4x4_2(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )

class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)

class MultiAttrEncoder(nn.Module):
    def __init__(self):
        super(MultiAttrEncoder, self).__init__()
        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)

        self.deconv1 = deconv4x4(1024, 1024)
        self.deconv2 = deconv4x4(2048, 512)
        self.deconv3 = deconv4x4(1024, 256)
        self.deconv4 = deconv4x4(512, 128)
        self.deconv5 = deconv4x4(256, 64)
        self.deconv6 = deconv4x4(128, 32)

        self.apply(weight_init)

    def forward(self, Xt):
        # 32*128*128
        f1 = self.conv1(Xt)
        # 64*64*64
        f2 = self.conv2(f1)
        # 128*32*32
        f3 = self.conv3(f2)
        # 256*16*16
        f4 = self.conv4(f3)
        # 512*8*8
        f5 = self.conv5(f4)
        # 1024*4*4
        f6 = self.conv6(f5)
        # 1024*2*2
        z_attr1 = self.conv7(f6)

        # # 2048*4*4
        # z_att2 = torch.cat((self.deconv1(z_att1), f6), dim=1)
        # # 1024*8*8
        # z_att3 = torch.cat((self.deconv2(z_att2), f5), dim=1)
        # # 512*16*16
        # z_att4 = torch.cat((self.deconv3(z_att3), f4), dim=1)
        # # 256*32*32
        # z_att5 = torch.cat((self.deconv4(z_att4), f3), dim=1)
        # # 128*64*64
        # z_att6 = torch.cat((self.deconv5(z_att5), f2), dim=1)
        # # 64*128*128
        # z_att7 = torch.cat((self.deconv6(z_att6), f1), dim=1)
        # # 64*256*256
        # z_att8 = F.interpolate(z_att7, scale_factor=2, mode='bilinear', align_corners=True)
        z_attr2 = self.deconv1(z_attr1, f6)
        z_attr3 = self.deconv2(z_attr2, f5)
        z_attr4 = self.deconv3(z_attr3, f4)
        z_attr5 = self.deconv4(z_attr4, f3)
        z_attr6 = self.deconv5(z_attr5, f2)
        z_attr7 = self.deconv6(z_attr6, f1)
        z_attr8 = F.interpolate(z_attr7, scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8
        # return z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8

class AAD_generator(nn.Module):
    def __init__(self, c_id=256):
        super(AAD_generator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id)
        self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk4 = AAD_ResBlk(1024, 512, 512, c_id)
        self.AADBlk5 = AAD_ResBlk(512, 256, 256, c_id)
        self.AADBlk6 = AAD_ResBlk(256, 128, 128, c_id)
        self.AADBlk7 = AAD_ResBlk(128, 64, 64, c_id)
        self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id)

        self.apply(weight_init)
    
    def forward(self, z_attr, z_id):
        x = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        # print("--------{}---------".format(x.shape))
        x = F.interpolate(self.AADBlk1(x, z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk2(x, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk3(x, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk4(x, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk5(x, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk6(x, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(self.AADBlk7(x, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        y = self.AADBlk8(x, z_attr[7], z_id)
        return torch.tanh(y)

class AEI_Net(nn.Module):
    def __init__(self, c_id=256, c_x=3):
        super(AEI_Net, self).__init__()
        self.encoder = MultiAttrEncoder()
        self.generator = AAD_generator(c_id)
    
    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr
    
    def get_attr(self, X):
        # with torch.no_grad():
        return self.encoder(X)