import cv2
import torch
import numpy as np
from typing import Any
from functools import cached_property
from gym_hepha.envs.hepha import GymHepha
from lerobot.robots.hepha_follower.hepha_auto_controller.auto_controller import AutoController


@AutoController.register_subclass("rule_based_controller")
class RuleBasedController(AutoController):
    """
    Auto controller using rules based decision to interact with the Hepha environment.

    This class is registered under the name 'rule_based_controller' in the AutoController registry,
    enabling dynamic selection of this controller via its string identifier.

    Attributes:

    """

    def __init__(self, show_camera: bool = False):
        super().__init__()
        self.observation_width = 264
        self.observation_height = 264
        self.visualization_width = 160
        self.visualization_height = 160

        # ---------------------
        # Show camera (from base class)
        # ---------------------
        self.show_camera = show_camera  # Display camera feed during interaction

        # ---------------------
        # Create environment
        # ---------------------
        self.env = GymHepha(
            task_name="bucket_to_bin",
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            visualization_width=self.visualization_width,
            visualization_height=self.visualization_height
        )
        self.frame_index = 0
        self.done = False

    def get_cameras(self) -> dict[str, Any]:
        """
        Return a dictionary of available cameras. In this simulated environment,
        no physical cameras are available, so a simulated camera "cam_0" is returned with a value of None.

        Returns:
            dict[str, Any]: Dictionary mapping camera names to camera interfaces or None.
        """
        return {"top_view": None, "gripper_cam": None}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {"z_slide_1": float, "x_slide": float, "y_slide": float, "rotate_arm_1": float,
                "slide_gripper_finger_0": float,
                "bucket_pos_x": float, "bucket_pos_y": float, "bucket_pos_z": float,
                "bin_pos_x": float, "bin_pos_y": float, "bin_pos_z": float,
                "top_view": (self.observation_width, self.observation_height, 3),
                "gripper_cam": (self.observation_width, self.observation_height, 3)}

    @cached_property
    def action_features(self) -> dict[str, type]:

        return {"z_slide_1": float, "x_slide": float, "y_slide": float,
                "rotate_arm_1": float, "slide_gripper_finger_0": float
                }

    def reset(self):
        """
        Reset the environment.
        """
        obs, _ = self.env.reset()
        self.frame_index = 0
        self.done = False

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the environment with added Gaussian noise
        and scaled agent position.

        This method:
        - Scales joint positions to [-1, 1] using self.env.scale_agent_pos().
        - Adds Gaussian noise to scaled joint positions.
        - Clips joint positions to [-1, 1] after noise.
        - Adds Gaussian noise to both top-view and gripper camera images.

        Returns:
            dict[str, Any]: A dictionary containing:
                - "z_slide_1" (float): Scaled + noisy vertical position of the slide.
                - "x_slide" (float): Scaled + noisy X-axis position.
                - "y_slide" (float): Scaled + noisy Y-axis position.
                - "rotate_arm_1" (float): Scaled + noisy arm rotation.
                - "slide_gripper_finger_0" (float): Scaled + noisy gripper extension.
                - "gripper_cam" (np.ndarray): Noisy RGB image from gripper camera.
                - "top_view" (np.ndarray): Noisy RGB image from top camera.
        """
        obs = self.env.scale_obs(self.env.get_obs())

        agent_pos = np.array(obs["agent_pos"], dtype=np.float32)

        # Step 2: Add Gaussian noise
        noise_std_pos = 0.01
        noisy_agent_pos = agent_pos + np.random.normal(0, noise_std_pos, size=agent_pos.shape)

        # Step 3: Clip to [-1, 1]
        noisy_agent_pos = np.clip(noisy_agent_pos, -1.0, 1.0)

        # Process images
        gripper_cam = obs["pixels"][:, :, :3]
        top_view = obs["pixels"][:, :, 3:]

        noise_std_img = 10.0
        noisy_gripper_cam = np.clip(gripper_cam + np.random.normal(0, noise_std_img, gripper_cam.shape), 0, 255).astype(
            np.uint8)
        noisy_top_view = np.clip(top_view + np.random.normal(0, noise_std_img, top_view.shape), 0, 255).astype(np.uint8)

        bucket_pos = np.array(obs["bucket_pos"], dtype=np.float32)
        bin_pos = np.array(obs["bin_pos"], dtype=np.float32)

        return {
            "z_slide_1": noisy_agent_pos[0],
            "x_slide": noisy_agent_pos[1],
            "y_slide": noisy_agent_pos[2],
            "rotate_arm_1": noisy_agent_pos[3],
            "slide_gripper_finger_0": noisy_agent_pos[4],
            "bucket_pos_x": bucket_pos[0],
            "bucket_pos_y": bucket_pos[1],
            "bucket_pos_z": bucket_pos[2],
            "bin_pos_x": bin_pos[0],
            "bin_pos_y": bin_pos[1],
            "bin_pos_z": bin_pos[2],
            "gripper_cam": noisy_gripper_cam,
            "top_view": noisy_top_view
        }

    def get_action(self):
        """
        Compute the next action based on the current state of the environment.

        Returns:
            dict: Scaled action dictionary containing values for z_slide_1, x_slide, y_slide,
                  rotate_arm_1, and slide_gripper_finger_0, all in range [-1, 1].
        """
        # Get the next action for this frame index
        action = self.env.get_action(self.frame_index)
        self.frame_index += 1

        # Apply action in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = self.done or truncated

        # Update observation queues
        top_view_image = cv2.resize(obs["pixels"][:, :, :3], (self.observation_width, self.observation_height))
        gripper_cam_image = cv2.resize(obs["pixels"][:, :, 3:], (self.observation_width, self.observation_height))

        # Optional: Show agent's camera view
        if self.show_camera:
            top_bgr = cv2.cvtColor(top_view_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Top View (RGB)", top_bgr)

            gripper_bgr = cv2.cvtColor(gripper_cam_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Gripper Camera View (RGB)", gripper_bgr)

            key = cv2.waitKey(10)
            if key == ord('q'):
                self.done = True

        # Scale the action using env's method
        scaled_action = self.env.scale_action(action)

        # Ensure all action values are within [-1, 1]
        if not np.all(np.logical_and(scaled_action >= -1.0, scaled_action <= 1.0)):
            raise ValueError(f"Scaled action values must be in [-1, 1]. Received: {scaled_action}")

        return {
            "z_slide_1": scaled_action[0],
            "x_slide": scaled_action[1],
            "y_slide": scaled_action[2],
            "rotate_arm_1": scaled_action[3],
            "slide_gripper_finger_0": scaled_action[4],
        }

    def close(self):
        """
        Close the environment and any open windows to clean up resources.
        """
        if self.env is not None:
            self.env.close()
            self.env = None
        cv2.destroyAllWindows()
