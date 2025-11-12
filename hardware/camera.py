import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

import cv2
import logging
import pyKinectAzure.pyKinectAzure as pykinect

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        if not self.device_opened:
            raise RuntimeError("Device not connected")

        if self.kinect.device_get_capture(1000):
            depth_image = self.kinect.capture_get_depth_image()
            color_image = self.kinect.capture_get_color_image()

            if depth_image is None or color_image is None:
                return None

            # Convert to numpy arrays
            depth_img = np.array(depth_image, dtype=np.float32)
            color_img = np.array(color_image, dtype=np.uint8)

            # Convert BGRA → RGB
            if color_img.shape[2] == 4:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2RGB)

            # Resize depth to color size
            depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert depth from mm → meters
            depth_img *= 0.001

            # Force shape to (H, W, 1)
            if depth_img.ndim == 2:
                depth_img = depth_img[:, :, np.newaxis]
            elif depth_img.ndim == 3 and depth_img.shape[2] != 1:
                depth_img = depth_img[:, :, 0:1]

            return {
                'rgb': color_img,
                'aligned_depth': depth_img,
            }
        else:
            logger.warning("Timeout waiting for frame")
            return None

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()



logger = logging.getLogger(__name__)


if __name__ == '__main__':
    
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()

    while True:
        cam.plot_image_bundle()

