import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from AEI_net import *
from face_modules.mtcnn import *
import cv2
import PIL.Image as Image
import numpy as np
import glob
import time
import os
import shutil

output_path = './hearnet_data'
os.makedirs(output_path, exist_ok=True)

batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = AEI_Net(c_id=512).to(device)
G.eval()
G.load_state_dict(torch.load('../saved_models/G_latest.pth', map_location=torch.device('cpu')))

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('../face_modules/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_roots = ['../aligned']
all_lists = []
for data_root in data_roots:
    img_lists = glob.glob(data_root + '/*.*g')
    all_lists.append(img_lists)
all_lists = sum(all_lists, [])

# with torch.no_grad():
#     for idx in range(0, )