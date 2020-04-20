import torch
import torch.nn as nn
import torch.nn.functional as F

class AAD(nn.Module):
    def __init__(self, c_h, c_attr, c_id=256):
        super(AAD, self).__init__()
        self.c_h = c_h
        self.c_attr = c_attr
        self.c_id = c_id

        self.norm = nn.InstanceNorm2d(c_h, affine=False)
        self.conv1 = nn.Conv2d(c_attr, c_h, 1, 1)
        self.conv2 = nn.Conv2d(c_attr, c_h, 1, 1)
        self.fc1 = nn.Linear(c_id, c_h)
        self.fc2 = nn.Linear(c_id, c_h)

        self.conv_h = nn.Conv2d(c_h, 1, 1, 1)

    def forward(self, h_in, z_att, z_id):
        h_norm = self.norm(h_in)

        gamma_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)

        gamma_id = self.fc1(z_id).reshape(h_norm.shape[0], self.c_h, 1, 1).expand_as(h_norm)
        beta_id = self.fc2(z_id).reshape(h_norm.shape[0], self.c_h, 1, 1).expand_as(h_norm)

        M = torch.sigmoid(self.conv_h(h_norm))
        A = torch.mul(gamma_att, h_in) + beta_att
        I = torch.mul(gamma_id, h_in) + beta_id
        out = (torch.ones_like(M) - M) * A + M * I
        return out

class AAD_ResBlk(nn.Module):
    def __init__(self, c_in, c_out, c_att, c_id=256):
        super(AAD_ResBlk, self).__init__()
        self.c_in = c_in
        self.c_out = c_out

        self.AAD1 = AAD(c_in, c_att, c_id)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, padding=1, bias=False)
        self.AAD2 = AAD(c_in, c_att, c_id)
        self.conv2 = nn.Conv2d(c_in, c_out, 3, 1, padding=1, bias=False)

        if c_in != c_out:
            self.AAD3 = AAD(c_in, c_att, c_id)
            self.conv3 = nn.Conv2d(c_in, c_out, 3, 1, padding=1, bias=False)
    
    def forward(self, h, z_att, z_id):
        x = F.relu(self.AAD1(h, z_att, z_id))
        x = self.conv1(x)

        x = F.relu(self.AAD1(x, z_att, z_id))
        x = self.conv2(x)

        if self.c_in != self.c_out:
            h = F.relu(self.AAD3(h, z_att, z_id))
            h = self.conv3(h)
        x = x + h

        return x
