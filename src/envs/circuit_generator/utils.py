"""
Copyright (c) 2024 Michikuni Eguchi
Released under the MIT license
"""

import matplotlib.pyplot as plt
import math
import numpy as np

def circle(center_x, center_y, radius, start=0., end=2*np.pi, dl=0.1):
    """ Create circle matrix
    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        start (float): start angle
        end (float): end angle
    Returns:
        circle x : numpy.ndarray
        circle y : numpy.ndarray
    """

    diff = end - start

    arc_length = 2*np.pi*radius*(diff/(2*np.pi))
    n_point = round(arc_length/dl)

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i*diff/n_point + start))
        circle_ys.append(center_y + radius * np.sin(i*diff/n_point + start))

    return np.array(circle_xs), np.array(circle_ys)


def plot_arrow(ax, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    ax.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    ax.plot(x, y)


def plot_robot(x, y, yaw, robot_radius):  # pragma: no cover
    circle = plt.Circle((x, y), robot_radius)
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")