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

import warnings
warnings.filterwarnings("ignore")

from estimator import estimator

#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
maml_parameters_path = 'demo_weights/weights_maml'



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
ted_weights = torch.load(ted_parameters_path, map_location=torch.device('cpu'))
#if torch.cuda.device_count() == 1:
if next(iter(ted_weights.keys())).startswith('module.'):
   ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])


gaze_network.load_state_dict(ted_weights)

# Initialize monitor and frame processor

DIR_TO_SAVE = "data"
dir_list = os.listdir(DIR_TO_SAVE)


for dir in dir_list:
     print("Counting loss for ", dir)
     cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}

     path = os.path.join(DIR_TO_SAVE, dir)

     cam_calib = pickle.load(open(os.path.join("calib_cam%d.pkl" % (0)), "rb"))

     ipaths = sorted(glob.glob(os.path.join(path, 'img_*.jpg')))
     images = []
     for ipath in ipaths:
         img = cv2.imread(ipath)
         images.append(img)

     points_path = os.path.join(path, f"points.pickle")

     with open(points_path, 'rb') as file:
         points = pickle.load(file)

     est = estimator(cam_calib)

     errors = []

     for idx, image in enumerate(images):
         errors.append(est.estimate(device, gaze_network, image, points[idx]))

     mean_error = torch.mean(torch.FloatTensor(errors))

     print(f"Mean error for {dir} is {mean_error}")


