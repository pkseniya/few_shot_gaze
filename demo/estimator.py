#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
from subprocess import call
import numpy as np
from os import path
import pickle
import sys
import os
import torch

sys.path.append("ext/eth")
from undistorter import Undistorter
from KalmanFilter1D import Kalman1D

from face import face
from landmarks import landmarks
from head import PnPHeadPoseEstimator
from normalization import normalize
from frame_processor import frame_processer
import torch.nn.functional as F
sys.path.append("../src")
from losses import GazeAngularLoss



class estimator(frame_processer):


    def __init__(self, cam_calib):
        super().__init__(cam_calib)
    def estimate(self, device, gaze_network, img, g_t):

        img = self.undistorter.apply(img)

        # detect face
        face_location = face.detect(img,  scale=0.25, use_max='SIZE')

        if len(face_location) > 0:
            # use kalman filter to smooth bounding box position
            # assume work with complex numbers:
            output_tracked = self.kalman_filters[0].update(face_location[0] + 1j * face_location[1])
            face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
            output_tracked = self.kalman_filters[1].update(face_location[2] + 1j * face_location[3])
            face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

            # detect facial points
            pts = self.landmarks_detector.detect(face_location, img)
            # run Kalman filter on landmarks to smooth them
            for i in range(68):
                kalman_filters_landm_complex = self.kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
                pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

            # compute head pose
            fx, _, cx, _, fy, cy, _, _, _ = self.cam_calib['mtx'].flatten()
            camera_parameters = np.asarray([fx, fy, cx, cy])
            rvec, tvec = self.head_pose_estimator.fit_func(pts, camera_parameters)

            ######### GAZE PART #########

            # create normalized eye patch and gaze and head pose value,
            # if the ground truth point of regard is given
            head_pose = (rvec, tvec)
            por = np.zeros((3, 1))
            por[0] = g_t[0]
            por[1] = g_t[1]

            entry = {
                    'full_frame': img,
                    '3d_gaze_target': por,
                    'camera_parameters': camera_parameters,
                    'full_frame_size': (img.shape[0], img.shape[1]),
                    'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                            int(face_location[2] - face_location[0]),
                                            int(face_location[3] - face_location[1]))
                    }
            [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)


            def preprocess_image(image):
                ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

                image = np.transpose(image, [2, 0, 1])  # CxHxW
                image = 2.0 * image / 255.0 - 1
                return image

            # estimate the PoR using the gaze network
            processed_patch = preprocess_image(patch)
            processed_patch = processed_patch[np.newaxis, :, :, :]

            # Functions to calculate relative rotation matrices for gaze dir. and head pose
            def R_x(theta):
                sin_ = np.sin(theta)
                cos_ = np.cos(theta)
                return np.array([
                    [1., 0., 0.],
                    [0., cos_, -sin_],
                    [0., sin_, cos_]
                ]).astype(np.float32)

            def R_y(phi):
                sin_ = np.sin(phi)
                cos_ = np.cos(phi)
                return np.array([
                    [cos_, 0., sin_],
                    [0., 1., 0.],
                    [-sin_, 0., cos_]
                ]).astype(np.float32)

            def calculate_rotation_matrix(e):
                return np.matmul(R_y(e[1]), R_x(e[0]))

            R_head_a = calculate_rotation_matrix(h_n)
            R_gaze_a = np.zeros((1, 3, 3))
            input_dict = {
                'image_a': processed_patch,
                'gaze_a': g_n,
                'head_a': h_n,
                'R_gaze_a': R_gaze_a,
                'R_head_a': R_head_a,
            }

            # compute eye gaze and point of regard
            for k, v in input_dict.items():
                input_dict[k] = torch.FloatTensor(v).to(device).detach()

            gaze_network.eval()
            output_dict = gaze_network(input_dict)
            output = output_dict['gaze_a_hat']
            g_cnn = output.data.cpu().numpy()
            g_cnn = g_cnn.reshape(3, 1)
            g_cnn /= np.linalg.norm(g_cnn)


            def pitchyaw_to_vector(pitchyaw):
                vector = np.zeros((3, 1))
                vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
                vector[1, 0] = np.sin(pitchyaw[0])
                vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
                return vector

            g_vector = torch.FloatTensor(pitchyaw_to_vector(g_n))
            return nn_angular_distance(g_vector, torch.FloatTensor(g_cnn))


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(sim) * (180 / np.pi))
