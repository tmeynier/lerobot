import cv2
import torch
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
            observation_width=264,
            observation_height=264,
            visualization_width=160,
            visualization_height=160
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
                "slide_gripper_finger_0": float, "top_view": (self.observation_width, self.observation_height, 3),
                "gripper_cam": (self.observation_width, self.observation_height, 3)}

    @cached_property
    def action_features(self) -> dict[str, type]:

        return {"z_slide_1": float, "x_slide": float, "y_slide": float,
                "rotate_arm_1": float, "slide_gripper_finger_0": float,}

    def reset(self):
        """
        Reset the environment.
        """
        obs, _ = self.env.reset()
        self.frame_index = 0
        self.done = False

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the environment.

        This method extracts positional and visual data from the Hepha environment and
        returns it as a structured dictionary. The observation includes the robot's
        joint positions (e.g., slide, rotation, gripper) as well as images from the
        top-view and gripper cameras.

        Returns:
            dict[str, Any]: A dictionary containing:
                - "z_slide_1" (float): Vertical position of the slide.
                - "x_slide" (float): Horizontal position along the X-axis.
                - "y_slide" (float): Horizontal position along the Y-axis.
                - "rotate_arm_1" (float): Rotation angle of the arm.
                - "slide_gripper_finger_0" (float): Extension of the gripper finger.
                - "top_view" (np.ndarray): RGB image from the top-down camera.
                - "gripper_cam" (np.ndarray): RGB image from the gripper-mounted camera.
        """
        obs = self.env.get_obs()

        return {
            "z_slide_1": obs["agent_pos"][0],
            "x_slide": obs["agent_pos"][1],
            "y_slide": obs["agent_pos"][2],
            "rotate_arm_1": obs["agent_pos"][3],
            "slide_gripper_finger_0": obs["agent_pos"][4],
            "gripper_cam": obs["pixels"][:,:,:3],
            "top_view": obs["pixels"][:,:,3:]
        }

    def get_action(self):
        """
        Compute the next action based on the current state of the environment.

        Returns:
            dict: Action dictionary containing values for motor_0 and motor_1.
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
            # Show top view (RGB)
            top_bgr = cv2.cvtColor(top_view_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Top View (RGB)", top_bgr)

            gripper_bgr = cv2.cvtColor(gripper_cam_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Gripper Camera View (RGB)", gripper_bgr)

            key = cv2.waitKey(10)
            if key == ord('q'):
                self.done = True  # Allow early exit if 'q' is pressed

        # Format and return the action
        return {"z_slide_1": action[0], "x_slide": action[1],
                "y_slide": action[2], "rotate_arm_1": action[3], "slide_gripper_finger_0": action[4]}

    def close(self):
        """
        Close the environment and any open windows to clean up resources.
        """
        if self.env is not None:
            self.env.close()
            self.env = None
        cv2.destroyAllWindows()
