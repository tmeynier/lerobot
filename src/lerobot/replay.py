import cv2
import random
import fsspec
import numpy as np
import pandas as pd
from gym_hepha.envs.hepha import GymHepha

# Constants
DATASET_NAME = "tmeynier/test_hepha_record_22"
REPO_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main"
CHUNK_ID = 0
NUM_EPISODES = 300

# Action scaling bounds
ACTION_MIN = np.asarray([-0.225, -0.271, -0.175, -1.5708, -0.0333])
ACTION_MAX = np.asarray([ 0.225,  0.271,  0.175,  0.7854,  0.0333])

# Pick a random episode
episode_id = random.randint(0, NUM_EPISODES - 1)
print(f"[INFO] Selected episode {episode_id}")

# Build Parquet file URL
parquet_path = f"{REPO_URL}/data/chunk-{CHUNK_ID:03d}/episode_{episode_id:06d}.parquet"
print(f"[INFO] Loading parquet file: {parquet_path}")

# Load the episode
with fsspec.open(parquet_path) as f:
    df = pd.read_parquet(f)

# Denormalize actions
normalized_actions = np.stack(df["action"].tolist())
actions = 0.5 * (normalized_actions + 1.0) * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
print(f"[INFO] Loaded {len(actions)} actions.")

# Load videos from Hugging Face with fsspec and OpenCV
def open_hf_video(repo_url, path):
    """Returns cv2.VideoCapture object from a remote Hugging Face video URL."""
    video_url = f"{repo_url}/videos/chunk-{CHUNK_ID:03d}/{path}/episode_{episode_id:06d}.mp4"
    print(f"[INFO] Loading video from: {video_url}")
    file = fsspec.open(video_url).open()
    # Read video into bytes, write to buffer, open with OpenCV
    video_bytes = file.read()
    np_arr = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    return video

# Alternative: using local tempfile if cv2.VideoCapture can't handle streams directly
def video_capture_from_hf(repo_url, path):
    import tempfile
    video_url = f"{repo_url}/videos/chunk-{CHUNK_ID:03d}/{path}/episode_{episode_id:06d}.mp4"
    print(f"[INFO] Loading video: {video_url}")
    with fsspec.open(video_url, "rb") as f:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(f.read())
            return cv2.VideoCapture(tmp.name)

# Load both videos
gripper_video = video_capture_from_hf(REPO_URL, "observation.images.gripper_cam")
top_view_video = video_capture_from_hf(REPO_URL, "observation.images.top_view")

# Setup environment
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
    # Environment step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Environment frame
    env_top = obs["pixels"][:, :, :3]
    env_gripper = obs["pixels"][:, :, 3:]

    env_top_bgr = cv2.cvtColor(env_top, cv2.COLOR_RGB2BGR)
    env_gripper_bgr = cv2.cvtColor(env_gripper, cv2.COLOR_RGB2BGR)

    # Read video frame
    ret_top, top_video_frame = top_view_video.read()
    ret_gripper, gripper_video_frame = gripper_video.read()

    if not ret_top or not ret_gripper:
        print("[WARN] Video frame read failed.")
        break

    # Resize all frames to same size (e.g. 320x320 for uniform grid)
    target_size = (320, 320)
    env_top_bgr_resized = cv2.resize(env_top_bgr, target_size)
    env_gripper_bgr_resized = cv2.resize(env_gripper_bgr, target_size)
    top_video_resized = cv2.resize(top_video_frame, target_size)
    gripper_video_resized = cv2.resize(gripper_video_frame, target_size)

    # Stack frames into 2x2 grid
    top_row = np.hstack((env_top_bgr_resized, gripper_video_resized))
    bottom_row = np.hstack((env_gripper_bgr_resized, top_video_resized))
    combined_frame = np.vstack((top_row, bottom_row))

    # Display the single combined frame
    cv2.imshow("🧠 Replay Viewer - Env vs Video", combined_frame)

    key = cv2.waitKey(100)
    if key == ord('q'):
        print("[INFO] Quit early")
        break

    if done:
        print("[INFO] Episode ended")
        break

env.close()
gripper_video.release()
top_view_video.release()
cv2.destroyAllWindows()
