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

import cv2
import logging
from typing import Any
from ..robot import Robot
from functools import cached_property
from .config_hepha_follower import HephaFollowerConfig
from .hepha_auto_controller.auto_controller import AutoController

logger = logging.getLogger(__name__)


class HephaFollower(Robot):
    """
    Hepha Follower designed by Meynier Tristan.
    """

    config_class = HephaFollowerConfig
    name = "hepha_follower"

    def __init__(self, config: HephaFollowerConfig):
        super().__init__(config)
        self.show_camera = config.show_camera

        # Create the auto controller using the registered name and show_camera setting
        self.auto_controller = AutoController.create(
            config.auto_controller,
            show_camera=config.show_camera
        )
        self.cameras = self.auto_controller.get_cameras()

    def reset(self):
        self.auto_controller.reset()

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
        return self.auto_controller.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return self.auto_controller.action_features

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

        return self.auto_controller.get_observation()

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
        return self.auto_controller.get_action()

    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        self.auto_controller.close()
        cv2.destroyAllWindows()  # Close OpenCV windows
