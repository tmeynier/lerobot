#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any
from ..robot import Robot
from .config_hepha_follower import HephaFollowerConfig
import gymnasium as gym
import gym_pusht  # This registers the environment
import cv2
from safetensors.torch import load_file
from lerobot.policies.dot.configuration_dot import DOTConfig
from lerobot.policies.dot.modeling_dot import DOTPolicy
from collections import deque
import torch
import safetensors
import os

logger = logging.getLogger(__name__)


class HephaFollower(Robot):
    """
    Hepha Follower designed by Meynier Tristan.
    """

    config_class = HephaFollowerConfig
    name = "hepha_follower"

    def __init__(self, config: HephaFollowerConfig, show_camera: bool = True):
        super().__init__(config)
        self.cameras = ["cam"]
        self.show_camera = show_camera

        ################################# ENV SIMULATOR #################################

        # Parameters
        self.nb_obs = 3
        self.image_size = (96, 96)
        self.state_dim = 2
        self.batch_size = 1
        self.done = False
        #checkpoint_path = os.path.expanduser("~/.models/dot/pusht/model.safetensors")
        checkpoint_path = "../../checkpoints/tristan_meynier/dot_pusht_images_25_06_2025/checkpoints/last/pretrained_model/model.safetensors"

        # Initialize observation queues
        self.image_queue = deque(maxlen=self.nb_obs)
        self.state_queue = deque(maxlen=self.nb_obs)

        # ---------------------
        # Config
        # ---------------------

        self.dot_config = DOTConfig()

        dataset_stats = {
            "action": {
                "max": torch.tensor([
                    512.0,
                    512.0
                ], dtype=torch.float32, device=self.dot_config.device),
                "min": torch.tensor([
                    0.0,
                    0.0
                ], dtype=torch.float32, device=self.dot_config.device),
            },
            "observation.state": {
                "max": torch.tensor([
                    512.0,
                    512.0
                ], dtype=torch.float32, device=self.dot_config.device),
                "min": torch.tensor([
                    0.0,
                    0.0
                ], dtype=torch.float32, device=self.dot_config.device),
            },
            "observation.image": {
                "mean": torch.tensor([
                    [[0.485]],
                    [[0.456]],
                    [[0.406]]
                ], dtype=torch.float32, device=self.dot_config.device),
                "std": torch.tensor([
                    [[0.229]],
                    [[0.224]],
                    [[0.225]]
                ], dtype=torch.float32, device=self.dot_config.device)
            }
        }

        # ---------------------
        # Load model
        # ---------------------

        self.dot_model = DOTPolicy(self.dot_config, dataset_stats)
        # model = ACTPolicy(config, dataset_stats)
        safetensors.torch.load_model(self.dot_model, checkpoint_path, strict=False, device=self.dot_config.device)
        self.dot_model.to(self.dot_config.device)
        self.dot_model.eval()

        # ---------------------
        # PushT environment
        # ---------------------

        # Create environment
        self.env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
        self.reset()

    def reset(self):
        obs, _ = self.env.reset()
        self.done = False

        # Reset queues
        self.image_queue.clear()
        self.state_queue.clear()

        # Warm-up with initial observation repeated
        for _ in range(self.nb_obs):
            image = cv2.resize(obs["pixels"], self.image_size)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
            state = torch.tensor(obs["agent_pos"], dtype=torch.float32)

            self.image_queue.append(image)
            self.state_queue.append(state)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {"motor_0": float, "motor_1": float, "cam": (96, 96, 3)}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {"motor_0": float, "motor_1": float}

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the robot.

        Args:
            calibrate (bool): If True, automatically calibrate the robot after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
        """
        obs = self.env.get_obs()

        return {
            "motor_0": obs["agent_pos"][0],
            "motor_1": obs["agent_pos"][0],
            "cam": obs["pixels"],
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """

        ################################# COMPUTE ENV ACTION #################################

        # Prepare batch
        stacked_images = torch.stack(list(self.image_queue))  # (nb_obs, 3, 96, 96)
        stacked_states = torch.stack(list(self.state_queue))  # (nb_obs, 2)

        batch = {
            "observation.image": stacked_images.to(self.dot_config.device),
            "observation.state": stacked_states.to(self.dot_config.device),
        }

        with torch.no_grad():
            action = self.dot_model.select_action(batch).cpu().numpy()
        action = action[0]

        # Step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = self.done or truncated

        # Update queues
        image = cv2.resize(obs["pixels"], self.image_size)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        state = torch.tensor(obs["agent_pos"], dtype=torch.float32)

        self.image_queue.append(image_tensor)
        self.state_queue.append(state)

        # ---- SHOW IMAGE USING OpenCV ----
        bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        cv2.imshow("PushT Agent View", bgr_frame)
        key = cv2.waitKey(10)  # Wait 10ms for key press
        if key == ord('q'):
            self.done = True  # Quit early if user presses 'q'

        action = {"motor_0": action[0], "motor_1": action[1]}
        return action

    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        self.env.close()
        cv2.destroyAllWindows()  # Close OpenCV windows

