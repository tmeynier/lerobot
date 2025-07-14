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
from ..teleoperator import Teleoperator
from .config_hepha_leader import HephaLeaderConfig
from typing import Any
import random

logger = logging.getLogger(__name__)


class HephaLeader(Teleoperator):
    """
    Hepha Leader designed by Meynier Tristan.
    """

    config_class = HephaLeaderConfig
    name = "hepha_leader"

    def __init__(self, config: HephaLeaderConfig):
        super().__init__(config)

    @property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions produced by the teleoperator. Its
        structure (keys) should match the structure of what is returned by :pymeth:`get_action`. Values for
        the dict should be the type of the value if it's a simple value, e.g. `float` for single
        proprioceptive value (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {"motor_0": float, "motor_1": float}

    @property
    def feedback_features(self) -> dict:
        """
        A dictionary describing the structure and types of the feedback actions expected by the robot. Its
        structure (keys) should match the structure of what is passed to :pymeth:`send_feedback`. Values for
        the dict should be the type of the value if it's a simple value, e.g. `float` for single
        proprioceptive value (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """
        Whether the teleoperator is currently connected or not. If `False`, calling :pymeth:`get_action`
        or :pymeth:`send_feedback` should raise an error.
        """
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish communication with the teleoperator.

        Args:
            calibrate (bool): If True, automatically calibrate the teleoperator after connecting if it's not
                calibrated or needs calibration (this is hardware-dependant).
        """
        pass

    @property
    def is_calibrated(self) -> bool:
        """Whether the teleoperator is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the teleoperator if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the teleoperator.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Retrieve the current action from the teleoperator.

        Returns:
            dict[str, Any]: A flat dictionary representing the teleoperator's current actions. Its
                structure should match :pymeth:`observation_features`.
        """
        start = time.perf_counter()
        action = {"motor_0": int(random.uniform(0, 680)), "motor_1": int(random.uniform(0, 680))}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Send a feedback action command to the teleoperator.

        Args:
            feedback (dict[str, Any]): Dictionary representing the desired feedback. Its structure should match
                :pymeth:`feedback_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect from the teleoperator and perform any necessary cleanup."""
        pass
