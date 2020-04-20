import torch
import torch.optim as optim
import torch.nn.functional as F
from AEI_net import *
from HEAR_net import *
from face_modules.model import Backbone
from utils.dataset import *
from torch.utils.data import DataLoader
# from apex import amp
import time

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available=", torch.cuda.is_available())

# common parameters
batch_size = 16
lr = 4e-4
max_epoch = 2000
# optim_level = 'O1'
show_iter = 1
save_iter = 300
model_save_path = './saved_models/'

# just use(eval) AEI_NET & Backbone
G = AEI_Net(c_id=512).to(device)
G.eval()
G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=True)

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

# train HEAR
net = HEAR()
net.train()
net.to(device)

# optimaization
opt = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))

# net, opt = amp.initialize(net, opt, opt_level=optim_level)

# load pretrained model
try:
    net.load_state_dict(torch.load('./saved_models/HEAR_latest.pth', map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print(e)

# prepare dataset
# dataset = AugmentedOcclusions('../hearnet_data',
#                               ['../ego_hands_png'],
#                               ['../shapenet_png'], same_prob=0.5)
test_data = dataset(['./aligned'])
dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# define loss
MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


for epoch in range(max_epoch):
    # torch.cuda.empty_cache() 
    for iter, data in enumerate(dataloader):

        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        same_person = same_person.to(device)

        with torch.no_grad():
            Zid_s, _ = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            Zid_t, _ = arcface(F.interpolate(Xt[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

        opt.zero_grad()
        with torch.no_grad():
            Yst_1, _ = G(Xt, Zid_s)
            Ytt, _ = G(Xt, Zid_t)
        
        dYt = Xt - Ytt
        hear_input = torch.cat((Yst_1, dYt), dim=1)

        Yst = net(hear_input)
        # may not necessary
        Yst_aligned = Yst[:, :, 19:237, 19:237]
        Yst_id, _ = arcface(F.interpolate(Yst_aligned, [112, 112], mode='bilinear', align_corners=True))
        # L_id loss
        L_id =(1 - torch.cosine_similarity(Zid_s, Yst_id, dim=1)).mean()
        # L_chg loss
        L_chg = L1(Yst_1, Yst)
        # L_res loss
        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Yst - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
        
        loss = L_id + L_chg + L_rec
        # with amp.scale_loss(loss, opt) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        opt.step()
        
        if iter % show_iter == 0:
            print("Train Epoch:{}, iteration:{}, Loss{}\n Loss_id{}, Loss_chg:{}, Loss_rec:{}".format(
                epoch, iter, loss.item(), L_id.item(), L_chg.item(), L_rec.item()))
        if iter % save_iter == 0:
            torch.save(net.state_dict(), './saved_models/Hear_latest.pth')
