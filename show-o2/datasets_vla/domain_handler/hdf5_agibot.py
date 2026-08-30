# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, Iterable
import numpy as np
import h5py

from ..utils import quat_to_rotate6d
from .base import BaseHDF5Handler


class AGIBOTHDF5Handler(BaseHDF5Handler):
    """
    HDF5 counterpart of ``AGIBOTLeRobotHandler``.

    Expected HDF5 layout (mirrors LeRobot parquet fields):
      /actions/end/position        [T, 2, 3]  L/R xyz
      /actions/end/orientation     [T, 2, 4]  L/R quaternion
      /actions/effector/position   [T, 2]     L/R gripper (0=open, 1=close; binarized)
      /language_instruction        (via meta ``language_instruction_key``)
      images via meta ``observation_key`` (e.g. ``rgb_comb``)

    Output left/right: [T, 10] = xyz(3) + rot6d(6) + grip(1), grip binary 1=closed.
    """

    dataset_name = "AGIBOT-HDF5-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float, float]:
        freq, qdur_max, qdur_min = 30.0, 2.0, 1.0

        pos = np.asarray(f["actions"]["end"]["position"][()])       # [T,2,3]
        ori = np.asarray(f["actions"]["end"]["orientation"][()])    # [T,2,4]
        grip = np.asarray(f["actions"]["effector"]["position"][()], dtype=np.float32)  # [T,2]
        # AgiBot: 0=open, 1=close; binarize (1=closed).
        if grip.ndim == 1:
            grip = grip[:, None]
        if grip.shape[-1] == 1 and grip.shape[0] == pos.shape[0] * 2:
            grip = grip.reshape(pos.shape[0], 2)
        grip = (grip > 0.5).astype(np.float32)

        left = np.concatenate(
            [pos[:, 0], quat_to_rotate6d(ori[:, 0]), grip[:, :1]],
            axis=-1,
        )  # [T,10]
        right = np.concatenate(
            [pos[:, 1], quat_to_rotate6d(ori[:, 1]), grip[:, 1:2]],
            axis=-1,
        )  # [T,10]

        return left, right, None, None, freq, qdur_max, qdur_min

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        candidates = list(range(0, max(0, T_left - 24)))
        n_keep = min(len(candidates), T_left // 2)
        return np.random.choice(candidates, size=n_keep, replace=False)
