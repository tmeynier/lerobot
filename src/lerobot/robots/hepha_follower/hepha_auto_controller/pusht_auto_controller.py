import os
import cv2
import torch
import gym_pusht
from typing import Any
import gymnasium as gym
import safetensors.torch
from collections import deque
from functools import cached_property
from lerobot.policies.dot.modeling_dot import DOTPolicy
from lerobot.policies.dot.configuration_dot import DOTConfig
from lerobot.robots.hepha_follower.hepha_auto_controller.auto_controller import AutoController


@AutoController.register_subclass("pusht_auto_controller")
class PushtAutoController(AutoController):
    """
    Auto controller using a DOTPolicy model to interact with the PushT environment.

    This class is registered under the name 'pusht_auto_controller' in the AutoController registry,
    enabling dynamic selection of this controller via its string identifier.

    Attributes:
        nb_obs (int): Number of stacked observations for the policy.
        image_size (tuple): Size to which input images are resized.
        state_dim (int): Dimension of the agent state.
        batch_size (int): Batch size for the model inference.
        done (bool): Flag indicating if the current episode is done.
        image_queue (deque): Buffer storing past image observations.
        state_queue (deque): Buffer storing past state observations.
        dot_config (DOTConfig): Configuration object for the DOTPolicy.
        dot_model (DOTPolicy): The trained DOT policy model.
        env (gym.Env): The simulation environment.
        show_camera (bool): Whether to display the camera feed during interaction.
    """

    def __init__(self, show_camera: bool = False):
        super().__init__()

        # ---------------------
        # Parameters
        # ---------------------
        self.nb_obs = 3
        self.image_size = (96, 96)
        self.state_dim = 2
        self.batch_size = 1
        self.done = False

        # ---------------------
        # Show camera (from base class)
        # ---------------------
        self.show_camera = show_camera  # Display camera feed during interaction

        # ---------------------
        # Model checkpoint path (updated to your local .models directory)
        # ---------------------
        #checkpoint_path = os.path.expanduser(
        #    "~/.models/dot/pusht/model.safetensors"
        #)
        checkpoint_path = ("../../checkpoints/tristan_meynier/dot_pusht_images_25_06_2025/"
                           "checkpoints/last/pretrained_model/model.safetensors")

        # ---------------------
        # Queues
        # ---------------------
        self.image_queue = deque(maxlen=self.nb_obs)
        self.state_queue = deque(maxlen=self.nb_obs)

        # ---------------------
        # Config & Dataset Stats
        # ---------------------
        self.dot_config = DOTConfig()

        dataset_stats = {
            "action": {
                "max": torch.tensor([512.0, 512.0], dtype=torch.float32, device=self.dot_config.device),
                "min": torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.dot_config.device),
            },
            "observation.state": {
                "max": torch.tensor([512.0, 512.0], dtype=torch.float32, device=self.dot_config.device),
                "min": torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.dot_config.device),
            },
            "observation.image": {
                "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32, device=self.dot_config.device),
                "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32, device=self.dot_config.device)
            }
        }

        # ---------------------
        # Load model
        # ---------------------
        self.dot_model = DOTPolicy(self.dot_config, dataset_stats)
        safetensors.torch.load_model(self.dot_model, checkpoint_path, strict=False, device=self.dot_config.device)
        self.dot_model.to(self.dot_config.device)
        self.dot_model.eval()

        # ---------------------
        # Create environment
        # ---------------------
        self.env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")

        # Initial reset
        self.reset()

    def get_cameras(self) -> dict[str, Any]:
        """
        Return a dictionary of available cameras. In this simulated environment,
        no physical cameras are available, so a simulated camera "cam_0" is returned with a value of None.

        Returns:
            dict[str, Any]: Dictionary mapping camera names to camera interfaces or None.
        """
        return {"cam_0": None}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive
            value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {"motor_0": float, "motor_1": float, "cam_0": (96, 96, 3)}

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

    def reset(self):
        """
        Reset the environment and internal observation queues.
        """
        obs, _ = self.env.reset()
        self.done = False

        # Clear old observations
        self.image_queue.clear()
        self.state_queue.clear()

        # Fill queues with initial repeated observation
        for _ in range(self.nb_obs):
            image = cv2.resize(obs["pixels"], self.image_size)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            state = torch.tensor(obs["agent_pos"], dtype=torch.float32)

            self.image_queue.append(image)
            self.state_queue.append(state)

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current sensory observation from the robot environment.

        This method returns a flattened dictionary containing observation
        data, such as motor positions and camera pixels. These
        observations are typically used to build a temporal stack for model inference
        in vision-based policy controllers like PushT.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory
                            state, including:
                              - "motor_0" (float): Position of motor 0.
                              - "motor_1" (float): Position of motor 1.
                              - "cam" (np.ndarray): RGB image from the robot's camera.
        """
        obs = self.env.get_obs()

        return {
            "motor_0": obs["agent_pos"][0],
            "motor_1": obs["agent_pos"][0],
            "cam_0": obs["pixels"],
        }

    def get_action(self):
        """
        Compute the next action based on the current state of the environment.

        Returns:
            dict: Action dictionary containing values for motor_0 and motor_1.
        """
        # Stack recent image and state observations
        stacked_images = torch.stack(list(self.image_queue))  # Shape: (nb_obs, 3, H, W)
        stacked_states = torch.stack(list(self.state_queue))  # Shape: (nb_obs, state_dim)

        batch = {
            "observation.image": stacked_images.to(self.dot_config.device),
            "observation.state": stacked_states.to(self.dot_config.device),
        }

        # Model inference
        with torch.no_grad():
            action = self.dot_model.select_action(batch).cpu().numpy()
        action = action[0]  # Extract action from batch

        # Apply action in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = self.done or truncated

        # Update observation queues
        image = cv2.resize(obs["pixels"], self.image_size)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        state = torch.tensor(obs["agent_pos"], dtype=torch.float32)

        self.image_queue.append(image_tensor)
        self.state_queue.append(state)

        # Optional: Show agent's camera view
        if self.show_camera:
            bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("PushT Agent View", bgr_frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                self.done = True  # Allow early exit if 'q' is pressed

        # Format and return the action
        return {"motor_0": action[0], "motor_1": action[1]}

    def close(self):
        """
        Close the environment and any open windows to clean up resources.
        """
        if self.env is not None:
            self.env.close()
            self.env = None
        cv2.destroyAllWindows()
