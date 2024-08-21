"""
Michikuni Eguchi, 2024.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from envs.circuit_generator.utils import plot_arrow

def interpolate_path(path, DL):
    # 経路の長さを計算
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    
    # NaNを除外
    if np.isnan(cumulative_distances).any():
        cumulative_distances = np.nan_to_num(cumulative_distances, nan=0.0)
    
    # 新しい点を生成するための新しい距離
    num_points = int(cumulative_distances[-1] // DL) + 1
    new_distances = np.linspace(0, cumulative_distances[-1], num_points)
    
    # 線形補間を使って新しい点を生成
    new_x = np.interp(new_distances, cumulative_distances, path[:, 0])
    new_y = np.interp(new_distances, cumulative_distances, path[:, 1])
    
    return np.column_stack((new_x, new_y))

def make_csv_paths(csv_file, DL=0.1, offset=True):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # オフセットを計算
    if offset:
        x_offset = df['x_m'].mean()
        y_offset = df['y_m'].mean()
    else:
        x_offset = 0
        y_offset = 0
    
    # circuit_pathを作成
    circuit_path = df[['x_m', 'y_m']].to_numpy()
    circuit_path[:, 0] -= x_offset
    circuit_path[:, 1] -= y_offset
    
    # パスの初期化
    right_path = []
    left_path = []
    
    for i in range(len(df)):
        x_m = df.loc[i, 'x_m'] - x_offset
        y_m = df.loc[i, 'y_m'] - y_offset
        w_tr_right_m = df.loc[i, 'w_tr_right_m']
        w_tr_left_m = df.loc[i, 'w_tr_left_m']
        
        if i > 0:
            prev_x_m = df.loc[i - 1, 'x_m'] - x_offset
            prev_y_m = df.loc[i - 1, 'y_m'] - y_offset
        else:
            prev_x_m = df.loc[len(df) - 1, 'x_m'] - x_offset
            prev_y_m = df.loc[len(df) - 1, 'y_m'] - y_offset
        
        # ベクトルの計算
        direction_vector = np.array([x_m - prev_x_m, y_m - prev_y_m])
        norm = np.linalg.norm(direction_vector)
        if norm != 0:
            direction_vector = direction_vector / norm
        else:
            direction_vector = np.array([1, 0])  # ゼロの場合のデフォルト方向ベクトル
        
        # 右方向と左方向のベクトル計算
        right_vector = np.array([-direction_vector[1], direction_vector[0]])
        left_vector = np.array([direction_vector[1], -direction_vector[0]])
        
        # 新しい右方向と左方向の点の計算
        right_x_m = x_m + w_tr_right_m * right_vector[0]
        right_y_m = y_m + w_tr_right_m * right_vector[1]
        left_x_m = x_m + w_tr_left_m * left_vector[0]
        left_y_m = y_m + w_tr_left_m * left_vector[1]
        
        # データリストに追加
        right_path.append([right_x_m, right_y_m])
        left_path.append([left_x_m, left_y_m])
    
    # NumPy配列に変換
    right_path = np.array(right_path)
    left_path = np.array(left_path)
    
    # 経路の補完
    circuit_path = interpolate_path(circuit_path, DL)
    right_path = interpolate_path(right_path, DL)
    left_path = interpolate_path(left_path, DL)
    
    # 角度と曲率の計算
    def calculate_angles(path):
        # path の最初と最後の点を利用して最初のベクトルを計算
        initial_direction_vector = np.array([path[0, 0] - path[-1, 0], path[0, 1] - path[-1, 1]])
        norm = np.linalg.norm(initial_direction_vector)
        if norm != 0:
            initial_direction_vector = initial_direction_vector / norm
        else:
            initial_direction_vector = np.array([1, 0])  # ゼロの場合のデフォルト方向ベクトル
        
        # 最初の角度を計算
        initial_angle = np.arctan2(initial_direction_vector[1], initial_direction_vector[0])
        
        road_diff = path[1:] - path[:-1]
        road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0])
        road_angle = np.concatenate(([initial_angle], road_angle))
        
        road = np.concatenate((path, road_angle[:, np.newaxis]), axis=1)
        
        return road
    
    circuit_path = calculate_angles(circuit_path)
    right_path = calculate_angles(right_path)
    left_path = calculate_angles(left_path)
    
    return circuit_path, right_path, left_path

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

    road, right_lane, left_lane = make_csv_paths('src/envs/circuit_generator/circuit.csv')

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