import cv2
import torch
import safetensors
import numpy as np
from collections import deque
from safetensors.torch import safe_open
from gym_hepha.envs.hepha import GymHepha
from lerobot.policies.dot.modeling_dot import DOTPolicy
from lerobot.policies.dot.configuration_dot import DOTConfig

# Parameters
nb_obs = 3
image_size = (264, 264)
state_dim = 5
batch_size = 1

# Initialize observation queues
top_view_image_queue = deque(maxlen=nb_obs)
gripper_cam_image_queue = deque(maxlen=nb_obs)
state_queue = deque(maxlen=nb_obs)

# ---------------------
# Config
# ---------------------
# Normalize action from [-1, 1] to [min, max]
ACTION_MIN = np.asarray([-0.225, -0.271, -0.175, -1.5708, -0.0333])
ACTION_MAX = np.asarray([ 0.225,  0.271,  0.175,  0.7854,  0.0333])

CHECKPOINT_PATH = "../../../checkpoints/tristan_meynier/dot_hepha_images_21_07_2025/checkpoints/last/pretrained_model/model.safetensors"


config = DOTConfig()

dataset_stats = {
    'action': {
        'min': torch.tensor([-0.22499999, -0.271, -0.175, -1.57079637, -0.03333334],
                            dtype=torch.float32, device=config.device),
        'max': torch.tensor([0.22499999, 0.271, 0.175, 0.00920367, 0.02666667],
                            dtype=torch.float32, device=config.device),
        'mean': torch.tensor([0.14123121, -0.00396572, 0.14933296, -0.67251203, -0.00263111],
                            dtype=torch.float32, device=config.device),
        'std': torch.tensor([0.10256309, 0.13619233, 0.07710076, 0.58981638, 0.02509085],
                            dtype=torch.float32, device=config.device),
    },
    'observation.state': {
        'min': torch.tensor([-0.22504972, -0.27100238, -0.16161282, -1.57109308, -0.03333334],
                            dtype=torch.float32, device=config.device),
        'max': torch.tensor([0.22500309, 0.27100295, 0.18204679, 0.00084918, 0.02664405],
                            dtype=torch.float32, device=config.device),
        'mean': torch.tensor([0.14117881, -0.00396553, 0.15004326, -0.67699052, -0.0028281],
                            dtype=torch.float32, device=config.device),
        'std': torch.tensor([0.10254265, 0.13619192, 0.07300066, 0.58604096, 0.02507234],
                            dtype=torch.float32, device=config.device),
    },
    'observation.images.gripper_cam': {
        'min': torch.tensor([[[0.]], [[0.]], [[0.]]], dtype=torch.float32, device=config.device),
        'max': torch.tensor([[[1.]], [[1.]], [[1.]]], dtype=torch.float32, device=config.device),
        'mean': torch.tensor([[[0.4850]], [[0.4560]], [[0.4060]]], dtype=torch.float32, device=config.device),
        'std': torch.tensor([[[0.2290]], [[0.2240]], [[0.2250]]], dtype=torch.float32, device=config.device),
    },
    'observation.images.top_view': {
        'min': torch.tensor([[[0.]], [[0.]], [[0.]]], dtype=torch.float32, device=config.device),
        'max': torch.tensor([[[1.]], [[1.]], [[1.]]], dtype=torch.float32, device=config.device),
        'mean': torch.tensor([[[0.4850]], [[0.4560]], [[0.4060]]], dtype=torch.float32, device=config.device),
        'std': torch.tensor([[[0.2290]], [[0.2240]], [[0.2250]]], dtype=torch.float32, device=config.device),
    }
}

# ---------------------
# Load model
# ---------------------

model = DOTPolicy(config, dataset_stats)
safetensors.torch.load_model(model, CHECKPOINT_PATH, strict=False, device=config.device)
model.to(config.device)
model.eval()

# ---------------------
# Gym Hepha environment
# ---------------------

# Create environment
env = GymHepha(
    task_name="bucket_to_bin",
    observation_width=264,
    observation_height=264,
    visualization_width=160,
    visualization_height=160
)
observation, info = env.reset()

# Main loop
for ep in range(5):  # Run 5 episodes
    obs, _ = env.reset()
    done = False
    total_reward = 0

    # Reset queues
    top_view_image_queue.clear()
    gripper_cam_image_queue.clear()
    state_queue.clear()

    # Warm-up with initial observation repeated
    for _ in range(nb_obs):
        top_view_image = cv2.resize(obs["pixels"][:,:,:3], image_size)
        gripper_cam_image = cv2.resize(obs["pixels"][:,:,3:], image_size)

        top_view_image = torch.tensor(top_view_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
        gripper_cam_image = torch.tensor(gripper_cam_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)

        state = torch.tensor(obs["agent_pos"][:5], dtype=torch.float32)

        top_view_image_queue.append(top_view_image)
        gripper_cam_image_queue.append(gripper_cam_image)

        state_queue.append(state)

    while not done:
        # Prepare batch
        stacked_top_view_images = torch.stack(list(top_view_image_queue))  # (nb_obs, 3, 96, 96)
        gripper_cam_view_images = torch.stack(list(gripper_cam_image_queue))  # (nb_obs, 3, 96, 96)

        stacked_states = torch.stack(list(state_queue))  # (nb_obs, 2)

        batch = {
            "observation.images.top_view": stacked_top_view_images.to(config.device),
            "observation.images.gripper_cam": gripper_cam_view_images.to(config.device),
            "observation.state": stacked_states.to(config.device),
        }

        with torch.no_grad():
            action = model.select_action(batch).cpu().numpy()
        action = action[0]

        # Denormalize
        action = 0.5 * (action + 1.0) * (ACTION_MAX - ACTION_MIN) + ACTION_MIN

        print("ACTION")
        print(action)

        # Step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = done or truncated
        total_reward += reward

        # Update queues
        top_view_image = cv2.resize(obs["pixels"][:,:,:3], image_size)
        gripper_cam_image = cv2.resize(obs["pixels"][:,:,3:], image_size)

        top_view_image = torch.tensor(top_view_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
        gripper_cam_image = torch.tensor(gripper_cam_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)

        state = torch.tensor(obs["agent_pos"][:5], dtype=torch.float32)

        top_view_image_queue.append(top_view_image)
        gripper_cam_image_queue.append(gripper_cam_image)

        state_queue.append(state)

        # ---- SHOW IMAGE USING OpenCV ----
        top_view_bgr_frame = top_view_image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        top_view_bgr_frame = (top_view_bgr_frame * 255).astype(np.uint8)  # Scale and convert to uint8
        top_view_bgr_frame = cv2.cvtColor(top_view_bgr_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Top View", top_view_bgr_frame)

        gripper_cam_bgr_frame = gripper_cam_image.permute(1, 2, 0).cpu().numpy()
        gripper_cam_bgr_frame = (gripper_cam_bgr_frame * 255).astype(np.uint8)
        gripper_cam_bgr_frame = cv2.cvtColor(gripper_cam_bgr_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Gripper Cam", gripper_cam_bgr_frame)

        key = cv2.waitKey(10)  # Wait 10ms for key press
        if key == ord('q'):
            done = True  # Quit early if user presses 'q'

env.close()
cv2.destroyAllWindows()
