import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    plt.pause(0.1)
    fig.canvas.draw()

    if save:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.savefig('results/{}.png'.format(time))


import os
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset_processing.grasp import detect_grasps
from datetime import datetime

def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None, save_folder='results', idx=0):
    """
    Save network output to disk (avoids opening GUI windows).
    """
    os.makedirs(save_folder, exist_ok=True)
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    # Save RGB
    if rgb_img is not None:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(rgb_img.astype(np.uint8))
        plt.axis('off')
        fig.savefig(os.path.join(save_folder, f'rgb_{idx:03d}.png'))
        plt.close(fig)

    # Save Depth
    if depth_img is not None:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(depth_img, cmap='gray')
        plt.axis('off')
        fig.savefig(os.path.join(save_folder, f'depth_{idx:03d}.png'))
        plt.close(fig)

    # Save Grasp overlay
    fig = plt.figure(figsize=(6,6))
    plt.imshow(rgb_img.astype(np.uint8))
    for g in gs:
        g.plot(plt.gca())
    plt.axis('off')
    fig.savefig(os.path.join(save_folder, f'grasp_{idx:03d}.png'))
    plt.close(fig)

    # Save Grasp quality
    fig = plt.figure(figsize=(6,6))
    plt.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    fig.savefig(os.path.join(save_folder, f'quality_{idx:03d}.png'))
    plt.close(fig)

    # Save Grasp angle
    fig = plt.figure(figsize=(6,6))
    plt.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
    plt.axis('off')
    fig.savefig(os.path.join(save_folder, f'angle_{idx:03d}.png'))
    plt.close(fig)

    # Save Grasp width
    if grasp_width_img is not None:
        fig = plt.figure(figsize=(6,6))
        plt.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
        plt.axis('off')
        fig.savefig(os.path.join(save_folder, f'width_{idx:03d}.png'))
        plt.close(fig)

