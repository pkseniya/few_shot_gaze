#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------
import warnings

try:
    import gi.repository
    gi.require_version('Gdk', '3.0')
    from gi.repository import Gdk
    gi_flag = True
except ModuleNotFoundError:
    gi_flag = False
    warnings.warn("No module named 'gi', default monitor settings: 596.7*335.6mm, 2560px*1440", Warning)

import numpy as np


class Monitor:

    def __init__(self):
        if gi_flag:
            display = Gdk.Display.get_default()
            screen = display.get_default_screen()
            default_screen = screen.get_default()
            num = default_screen.get_number()
            self.h_mm = default_screen.get_monitor_height_mm(num)
            self.w_mm = default_screen.get_monitor_width_mm(num)

            self.h_pixels = default_screen.get_height()
            self.w_pixels = default_screen.get_width()
        else:
            self.h_mm = 335.6
            self.w_mm = 596.7
            self.h_pixels = 1440 
            self.w_pixels = 2560 
        self.y_shift = 50

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = self.y_shift + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - self.y_shift) * self.h_pixels / self.h_mm)

        return x_mon_pixel, y_mon_pixel
