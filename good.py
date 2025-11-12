import argparse
import logging
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import pykinect_azure as pykinect

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results
from hardware.device import get_device

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate grasp network with Azure Kinect')
    parser.add_argument('--network', type=str, default='saved_data/cornell_rgbd_iou_0.96',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Initialize Azure Kinect: SET TO 30 FPS
    # ------------------------------------------------------------
    logging.info("Initializing Azure Kinect...")
    pykinect.initialize_libraries()

    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # 🎯 FPS CHANGE: Set to 30 FPS for maximum speed
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30 
    
    device = pykinect.start_device(config=device_config)
    logging.info("Azure Kinect started successfully at 30 FPS.")

    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # ------------------------------------------------------------
    # Load Network
    # ------------------------------------------------------------
    logging.info("Loading model...")
    net = torch.load(args.network, map_location='cpu')
    device_torch = get_device(args.force_cpu)
    net = net.to(device_torch)
    net.eval()
    logging.info("Model loaded successfully.")

    # ------------------------------------------------------------
    # Visualization setup: ONLY ONE WINDOW (Matplotlib)
    # ------------------------------------------------------------
    plt.ion() 
    fig = plt.figure(figsize=(10, 10))

    try:
        while True:
            capture = device.update()
            ret_rgb, color_image = capture.get_color_image()
            ret_depth, depth_image = capture.get_transformed_depth_image()

            if not ret_rgb or not ret_depth:
                continue

            # Convert BGRA → RGB for consistency
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            
            # --- Data Preprocessing ---
            depth = depth_image.astype(np.float32)
            depth *= 0.001
            if depth.ndim == 2:
                depth = np.expand_dims(depth, axis=2)
            # --------------------------
            
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)

            with torch.no_grad():
                xc = x.to(device_torch)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(
                    pred['pos'], pred['cos'], pred['sin'], pred['width']
                )

            # ------------------------------------------------------------
            # Visualize result
            # ------------------------------------------------------------
            plot_results(fig=fig,
                         rgb_img=cam_data.get_rgb(rgb, False),
                         depth_img=np.squeeze(cam_data.get_depth(depth)),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            
            # 🎯 DELAY REMOVAL: Removed plt.pause(0.01) to maximize speed.
            
            # Check for 'q' key press without blocking the loop
            if plt.waitforbuttonpress(timeout=0.001) and plt.get_current_fig_manager().key_press_handler.key == 'q':
                break

    finally:
        cv2.destroyAllWindows()
        plt.close(fig)
        logging.info("Session ended.")


if __name__ == '__main__':
    main()