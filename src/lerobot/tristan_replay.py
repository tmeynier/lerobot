import cv2
import random
import fsspec
import numpy as np
import pandas as pd
from datasets import load_dataset
from gym_hepha.envs.hepha import GymHepha


# Constants
DATASET_NAME = "tmeynier/test_hepha_record_22"
REPO_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main"
CHUNK_ID = 0
NUM_EPISODES = 300  # Adjust if needed, or list files from Hugging Face to determine this

# Pick a random episode
episode_id = random.randint(0, NUM_EPISODES - 1)
print(f"[INFO] Selected episode {episode_id}")

# Build URL to the Parquet file
parquet_path = f"{REPO_URL}/data/chunk-{CHUNK_ID:03d}/episode_{episode_id:06d}.parquet"
print(f"[INFO] Loading parquet file: {parquet_path}")

# Load Parquet from Hugging Face repo
with fsspec.open(parquet_path) as f:
    df = pd.read_parquet(f)

# Extract actions as NumPy array
actions = np.stack(df["action"].tolist())
print("ACTIONS")
print(actions)
print(f"[INFO] Loaded {len(actions)} actions.")

# ----------------------
# Setup environment
# ----------------------
env = GymHepha(
    task_name="bucket_to_bin",
    observation_width=264,
    observation_height=264,
    visualization_width=160,
    visualization_height=160
)

obs, _ = env.reset()
done = False

for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Display frames
    top_view = obs["pixels"][:, :, :3]
    gripper_cam = obs["pixels"][:, :, 3:]

    top_view_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
    gripper_cam_bgr = cv2.cvtColor(gripper_cam, cv2.COLOR_RGB2BGR)

    cv2.imshow("Replay - Top View", top_view_bgr)
    cv2.imshow("Replay - Gripper Cam", gripper_cam_bgr)

    key = cv2.waitKey(100)
    if key == ord('q'):
        print("[INFO] Quit early")
        break

    if done:
        print("[INFO] Episode ended")
        break

env.close()
cv2.destroyAllWindows()
