"""
Michikuni Eguchi, 2024.
"""

from math import ceil
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class LaneMap:
    """
    Lane map represented by a grid, with drivable area determined by distance from the lane centerline.
    """

    def __init__(
        self,
        lane: np.ndarray,
        lane_width: float,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.01,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ) -> None:
        """
        lane: Centerline of the lane in the form [[x_0, y_0, angle_0], ..., [x_n, y_n, angle_n]]
        lane_width: Width of the lane (drivable width)
        map_size: (width, height) [m], origin is at the center
        cell_size: (m)
        device: Torch device (e.g., cuda or cpu)
        dtype: Data type for the torch tensors
        """
        assert lane_width > 0
        assert len(lane.shape) == 2 and lane.shape[1] == 3

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # Initialize map and related attributes
        self.initialize_map(map_size, cell_size)

        # Populate the map with the lane's drivable area
        self.populate_map(lane, lane_width)

    def initialize_map(self, map_size: Tuple[int, int], cell_size: float) -> None:
        """
        Initialize the map and related attributes.
        """
        cell_map_dim = [ceil(map_size[0] / cell_size), ceil(map_size[1] / cell_size)]
        self._map = np.ones(cell_map_dim)
        self._cell_size = cell_size
        self._cell_map_origin = np.array([cell_map_dim[0] // 2, cell_map_dim[1] // 2])

        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(
            self._device, self._dtype
        )

        # Calculate the limits of the map in real-world coordinates
        self.x_lim = [-map_size[0] / 2, map_size[0] / 2]
        self.y_lim = [-map_size[1] / 2, map_size[1] / 2]

    def populate_map(self, lane: np.ndarray, lane_width: float) -> None:
        """
        Populate the map with the lane's drivable area.
        """
        # Mark the centerline on the map
        for point in lane:
            x, y, _ = point
            cell_x = int(round(x / self._cell_size)) + self._cell_map_origin[0]
            cell_y = int(round(y / self._cell_size)) + self._cell_map_origin[1]
            if 0 <= cell_x < self._map.shape[0] and 0 <= cell_y < self._map.shape[1]:
                self._map[cell_x, cell_y] = 0

        # Apply distance transform and update map
        distance_map = distance_transform_edt(self._map)
        max_distance = (lane_width / 2) / self._cell_size
        self._map = np.where(distance_map <= max_distance, 0, 1)

        # Convert the numpy map to torch tensor
        self._map_torch = torch.tensor(
            self._map, device=self._device, dtype=self._dtype
        )

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check collision in a batch of trajectories.
        :param x: Tensor of shape (batch_size, traj_length, position_dim).
        :return: collsion costs on the trajectories.
        """
        assert self._map_torch is not None
        if x.device != self._device or x.dtype != self._dtype:
            x = x.to(self._device, self._dtype)

        # project to cell map
        x_occ = (x / self._cell_size) + self._torch_cell_map_origin
        x_occ = torch.round(x_occ).long().to(self._device)

        # deal with out of bound
        is_out_of_bound = torch.logical_or(
            torch.logical_or(
                x_occ[..., 0] < 0, x_occ[..., 0] >= self._map_torch.shape[0]
            ),
            torch.logical_or(
                x_occ[..., 1] < 0, x_occ[..., 1] >= self._map_torch.shape[1]
            ),
        )
        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._map_torch.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._map_torch.shape[1] - 1)

        # collision check
        collisions = self._map_torch[x_occ[..., 0], x_occ[..., 1]]

        # out of bound cost
        collisions[is_out_of_bound] = 1.0

        return collisions

    def render_occupancy(self, ax, cmap="binary") -> None:
        extent = [self.x_lim[0], self.x_lim[1], self.y_lim[0], self.y_lim[1]]
        ax.imshow(self._map.T, cmap=cmap, origin="lower", extent=extent)
