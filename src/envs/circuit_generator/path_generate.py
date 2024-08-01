"""
Michikuni Eguchi, 2024.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs.circuit_generator.utils import plot_arrow, circle

def make_track(circle_radius, linelength, dl):
    """ make track
    Input parameters:
        circle_radius (float): circle radius
        linelength (float): line length

    Returns:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle, curvature
    """
    line_points = round(linelength/dl)

    line = np.linspace(-linelength/2, linelength/2, num=line_points+1, endpoint=False)[1:]
    line_1 = np.stack((line, np.zeros(line_points)), axis=1)
    line_2 = np.stack((line[::-1], np.zeros(line_points)+circle_radius*2.), axis=1)

    # circle
    circle_1_x, circle_1_y = circle(linelength/2., circle_radius,
                                    circle_radius, start=-np.pi/2., end=np.pi/2., dl=dl)
    circle_1 = np.stack((circle_1_x, circle_1_y), axis=1)

    circle_2_x, circle_2_y = circle(-linelength/2., circle_radius,
                                    circle_radius, start=np.pi/2., end=3*np.pi/2., dl=dl)
    circle_2 = np.stack((circle_2_x, circle_2_y), axis=1)

    road_pos = np.concatenate((line_1, circle_1, line_2, circle_2), axis=0)

    # calc road angle
    road_diff = road_pos[1:] - road_pos[:-1]
    road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0])
    road_angle = np.concatenate((np.zeros(1), road_angle))

    road = np.concatenate((road_pos, road_angle[:, np.newaxis]), axis=1)

    # calc road curvature
    road_curvature = calc_curvature_range_kutta(road[:, 0], road[:, 1])

    road = np.concatenate((road, np.array(road_curvature)[:, np.newaxis]), axis=1)

    # start offset
    road[:, 1] = road[:, 1] - circle_radius

    return road

def calc_curvature_range_kutta(x, y):
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    curvatures = [0.0, 0.0]
    for i in np.arange(2, len(x)-1):
        dx = (x[i+1] - x[i])/dists[i]
        dy = (y[i+1] - y[i])/dists[i]
        ddx = (x[i-2] - x[i-1] - x[i] + x[i+1])/(2*dists[i]**2)
        ddy = (y[i-2] - y[i-1] - y[i] + y[i+1])/(2*dists[i]**2)
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    curvatures.append(0.0)
    return curvatures

def make_side_lane(road, lane_width):
    """make_side_lane
    Input parameters:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle, curvature
        lane_width (float): width of the lane
    Output:
        right_lane (numpy.ndarray): shape(n_point, 3) x, y, angle
        left_lane  (numpy.ndarray): shape(n_point, 3) x, y, angle
    """
    right_lane_x = lane_width/2*np.cos(road[:,2]-np.pi/2) +road[:,0]
    right_lane_y = lane_width/2*np.sin(road[:,2]-np.pi/2) +road[:,1]
    right_lane_pos = np.stack((right_lane_x, right_lane_y), axis=1)

    left_lane_x = lane_width/2*np.cos(road[:,2]+np.pi/2) +road[:,0]
    left_lane_y = lane_width/2*np.sin(road[:,2]+np.pi/2) +road[:,1]
    left_lane_pos = np.stack((left_lane_x, left_lane_y), axis=1)

    road_angle = road[:,2]

    right_lane = np.concatenate((right_lane_pos, road_angle[:, np.newaxis]), axis=1)
    left_lane = np.concatenate((left_lane_pos, road_angle[:, np.newaxis]), axis=1)

    return right_lane, left_lane

if __name__ == '__main__':
    lane_width = 3
    circle_radius = 10
    linelength = 17

    # make track
    road = make_track(circle_radius, linelength, dl=0.1)
    right_lane, left_lane = make_side_lane(road, lane_width)
    #print(road)
    # track
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(road[:, 0], road[:, 1])
    ax1.plot(right_lane[:, 0], right_lane[:, 1])
    ax1.plot(left_lane[:, 0], left_lane[:, 1])
    ax1.set_title("track")
    # arrow plot
    for i in range(int(road[:,0].size/10)):
        plot_arrow(ax1, road[i*10, 0], road[i*10, 1], road[i*10, 2], 1, 0.5)    

    plt.show()