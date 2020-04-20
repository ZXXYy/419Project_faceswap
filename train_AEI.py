from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
# accelerate the training rate
# from apex import amp 
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from AEI_net import *
from Discriminator_net import *
from utils.dataset import dataset

print("cuda available=", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 1
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O1'
pretrained = True

# ----------------------load dataset------------------------------
train_data = dataset(['./aligned'])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

# ----------------------define networks---------------------------
# Indentity Encoder
arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

# AEI_net
G = AEI_Net(c_id=512).to(device)

# multi-scale discriminator for loss function
L = MultiScaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)

# -----------------------loss & optimizer-----------------------
opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_L = optim.Adam(L.parameters(), lr=lr_D, betas=(0, 0.999))

# G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
# L, opt_L = amp.initialize(L, opt_L, opt_level=optim_level)

# ---------------------functions used for train----------------------
def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()

def train(G, L, device, train_loader, opt_G, opt_L, epoch):
    
    for idx, data in enumerate(train_loader):
        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        # get image identity
        with torch.no_grad():
            z_id, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

        # ------------------train G----------------------------
        opt_G.zero_grad()
        Y, Xt_att = G(Xt, z_id)

        # compute loss
        Li = L(Y)
        L_adv = 0
        for li in Li:
            L_adv += hinge_loss(li[0], True)
        
        y_id, Y_feats = arcface(F.interpolate(Y[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(z_id, y_id, dim=1)).mean()

        Y_att = G.get_attr(Y)
        L_att = 0
        for i in range(len(Xt_att)):
            L_att += torch.mean(torch.pow(Xt_att[i] - Y_att[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_att /= 2.0

        # L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (batch_size)
        L_rec = 0
        for i in range(Xt.shape[0]):
            if same_person[i] == 1:
                loss_fn = torch.nn.MSELoss(reduction='mean')
                L_rec += loss_fn(Y[i], Xt[i])
        L_rec /= batch_size

        lossG = 1*L_adv + 10*L_att + 5*L_id + 10*L_rec
        # with amp.scale_loss(lossG, opt_G) as scaled_loss:
        #     scaled_loss.backward()
        # update
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 10)
        opt_G.step()

        # ----------------train L-----------------------------
        opt_L.zero_grad()
        fake_L = L(Y.detach())
        # compute loss
        loss_fake = 0
        for li in fake_L:
            loss_fake += hinge_loss(li[0], False)
        
        true_L = L(Xs)
        loss_true = 0
        for li in true_L:
            loss_true += hinge_loss(li[0], True)

        lossL = 0.5*(loss_true.mean() + loss_fake.mean())
        # with amp.scale_loss(lossL, opt_L) as scaled_loss:
        #    scaled_loss.backward()
        lossL.backward()
        opt_L.step()
        if idx % show_step == 0:
            print("Train Epoch:{}, iteration:{}, LossG:{}, LossL:{}".format(
                epoch, idx, lossG.item(), lossL.item()))
        if idx % 1000 == 0:
            torch.save(G.state_dict(), './saved_models/G_latest.pth')
            torch.save(L.state_dict(), './saved_models/D_latest.pth')

G.train()
L.train()
if pretrained:
    try:
        G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
        L.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')), strict=False)
    except Exception as e:
        print(e)

for epoch in range(max_epoch):
    train(G, L, device, train_dataloader, opt_G, opt_L, epoch)