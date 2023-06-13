# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Ray generator.
"""
from jaxtyping import Int
from torch import Tensor, nn

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        pose_optimizer: pose optimization module, for optimizing noisy camera intrinsics/extrinsics.
    """

    image_coords: Tensor
    pixel_offset: float

    def __init__(self, cameras: Cameras, pose_optimizer: CameraOptimizer, pixel_offset: float = 0.5) -> None:
        super().__init__()
        self.cameras = cameras
        self.pixel_offset = pixel_offset
        self.pose_optimizer = pose_optimizer
        self.register_buffer("image_coords", cameras.get_image_coords(pixel_offset), persistent=False)

    def forward(
        self, ray_indices: Int[Tensor, "num_rays 3"], resample: bool = False
    ) -> Union[RayBundle, Tuple[RayBundle, Float[Tensor, "num_rays 2"]]]:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        camera_opt_to_camera = self.pose_optimizer(c)

        ray_bundle_and_coords = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,
            resample=resample,
        )
        if resample:
            ray_bundle, new_coords = ray_bundle_and_coords
            new_coords = new_coords - self.pixel_offset
            return ray_bundle, new_coords
        return ray_bundle_and_coords
