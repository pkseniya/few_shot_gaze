#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import os
import pickle
import sys
import time
import warnings
from os import path
from subprocess import call

import cv2
import joblib
import numpy as np
import torch

warnings.filterwarnings("ignore")

from camera import cam_calibrate
from frame_processor import FrameProcesser
from monitor import Monitor
from parse_arguments import parse_arguments
from person_calibration import collect_data, fine_tune

sys.path.append("../src")
from models import DTED

#################################
# Start camera
#################################
if __name__ == '__main__':
    args = parse_arguments()

    cam_idx = 0

    # adjust these for your camera to get the best accuracy
    # call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
    # call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
    # call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

    cam_cap = cv2.VideoCapture(cam_idx)
    cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # calibrate camera
    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
    if path.exists(args.cam_calib):
        cam_calib = joblib.load(open(args.cam_calib, "rb"))
    else:
        print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
        print("Press s to save frame, c to continue, q to quit")
        cam_calibrate(cam_idx, cam_cap, cam_calib)

    #################################
    # Load gaze network
    #################################
    ted_parameters_path = args.ted_parameters_path
    maml_parameters_path = args.maml_parameters_path
    k = args.k

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create network
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
    ted_weights = torch.load(ted_parameters_path,map_location=torch.device('cpu'))
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

    #####################################

    # Load MAML MLP weights if available
    full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
    # print(full_maml_parameters_path)
    assert os.path.isfile(full_maml_parameters_path)
    print('> Loading: %s' % full_maml_parameters_path)
    maml_weights = torch.load(full_maml_parameters_path, map_location=torch.device('cpu'))
    ted_weights.update({  # rename to fit
        'gaze1.weight': maml_weights['layer01.weights'],
        'gaze1.bias':   maml_weights['layer01.bias'],
        'gaze2.weight': maml_weights['layer02.weights'],
        'gaze2.bias':   maml_weights['layer02.bias'],
    })
    gaze_network.load_state_dict(ted_weights)

    #################################
    # Personalize gaze network
    #################################

    # Initialize monitor and frame processor
    mon = Monitor()
    frame_processor = FrameProcesser(cam_calib)

    # collect person calibration data and fine-
    # tune gaze network
    subject = args.subject if args.subject else 'user'

    if not args.data_path:
        data = collect_data(cam_cap, mon, *args.num_points)
        joblib.dump(data, f'{subject}_calib_data.pkl')
    else:
        data = joblib.load(args.data_path)
    # adjust steps and lr for best results
    # To debug calibration, set show=True
    if not args.fine_tuning_path:
        gaze_network = fine_tune(subject, data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5, mode='train')
    else:
        gaze_network.load_state_dict(torch.load(args.fine_tuning_path))

    #################################
    # Run on live webcam feed and
    # show point of regard on screen
    #################################
    data = frame_processor.process(subject, cam_cap, mon, device, gaze_network, mode=args.mode)
