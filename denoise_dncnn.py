#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:35:44 2022

@author: electron
"""

from utils import utils_model
from utils import utils_image as util
import torch
from models.network_dncnn import DnCNN as net

def denoise_dncnn(img_L, x8=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'KAIR/model_zoo/dncnn_color_blind.pth'
    model = net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)
     
    if not x8:
        img_E = model(img_L)
    else:
        img_E = utils_model.test_mode(model, img_L, mode=3)
    
    img_E = util.tensor2uint(img_E)
    #util.imsave(img_E, 'out.png')
    return img_E