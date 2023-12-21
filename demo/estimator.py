import cv2
import numpy as np
import sys

from face import face

from normalization import normalize
from frame_processor import frame_processer


class estimator(frame_processer):

    def __init__(self, cam_calib):
        super().__init__(cam_calib)

    def preprocess(self, images, targets):
        data = {'image_a': [], 'gaze_a': [], 'head_a': [], 'R_gaze_a': [], 'R_head_a': []}

        for idx, img in enumerate(images):
            img = self.undistorter.apply(img)
            g_t = targets[idx]

            face_location = face.detect(img,  scale=0.25, use_max='SIZE')
            if len(face_location) == 0:
                continue

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
            patch, h_n, g_n, _, _, _ = normalize(entry, head_pose)

            def preprocess_image(image):
                ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                # cv2.imshow('processed patch', image)

                image = np.transpose(image, [2, 0, 1])  # CxHxW
                image = 2.0 * image / 255.0 - 1
                return image

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
            R_gaze_a = calculate_rotation_matrix(g_n)


            data['image_a'].append(processed_patch)
            data['gaze_a'].append(g_n)
            data['head_a'].append(h_n)
            data['R_gaze_a'].append(R_gaze_a)
            data['R_head_a'].append(R_head_a)

        return data  