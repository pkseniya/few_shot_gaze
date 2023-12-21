#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------


import cv2
import numpy as np
from os import path
import pickle
import sys
import torch
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

from estimator import estimator

sys.path.append("../src")
from losses import GazeAngularLoss

#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path, map_location=torch.device("cpu"))
if next(iter(ted_weights.keys())).startswith('module.'):
    ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])


gaze_network.load_state_dict(ted_weights)


DATASET_DIR = "data"
dir_list = os.listdir(DATASET_DIR)

for dir in dir_list:
    print("Evaluating", dir)
    
    path = os.path.join(DATASET_DIR, dir)

    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
    cam_calib = pickle.load(open(os.path.join(path, "calib_cam.pkl"), "rb"))

    ipaths = sorted(glob.glob(os.path.join(path, 'img_*.jpg')))
    images = []
    for ipath in ipaths:
        img = cv2.imread(ipath)
        images.append(img)

    points_path = os.path.join(path, f"points.pickle")

    with open(points_path, 'rb') as file:
        points = pickle.load(file)

    est = estimator(cam_calib)    
    data = est.preprocess(images, points)

    n = len(data['image_a'])

    _, c, h, w = data['image_a'][0].shape
    img = np.zeros((n, c, h, w))
    gaze_a = np.zeros((n, 2))
    head_a = np.zeros((n, 2))
    R_gaze_a = np.zeros((n, 3, 3))
    R_head_a = np.zeros((n, 3, 3))
    for i in range(n):
        img[i, :, :, :] = data['image_a'][i]
        gaze_a[i, :] = data['gaze_a'][i]
        head_a[i, :] = data['head_a'][i]
        R_gaze_a[i, :, :] = data['R_gaze_a'][i]
        R_head_a[i, :, :] = data['R_head_a'][i]

    input_dict_valid = {
        'image_a': img,
        'gaze_a': gaze_a,
        'head_a': head_a,
        'R_gaze_a': R_gaze_a,
        'R_head_a': R_head_a,
    }


    for k, v in input_dict_valid.items():
        input_dict_valid[k] = torch.FloatTensor(v).to(device).detach()


    gaze_network.eval()
    output_dict = gaze_network(input_dict_valid)
    
    loss = GazeAngularLoss()
    valid_loss = loss(input_dict_valid, output_dict).cpu()

    print('Mean angular error: %.2f' % (valid_loss.item()))
