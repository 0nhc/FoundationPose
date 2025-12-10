"""
This script is used to perform 6D pose tracking of the manipulated object
using the foundation models pose tracking server.
"""

import base64, io, numpy as np, cv2, requests

def encode_rgb_to_b64(rgb: np.ndarray) -> str:
    # rgb: HxWx3 uint8, RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")

def encode_npy_to_b64(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def encode_mesh_to_b64(mesh_path: str) -> str:
    with open(mesh_path, "rb") as f:
        mesh_bytes = f.read()
    return base64.b64encode(mesh_bytes).decode("ascii")


# Initial registration
K = np.load("outputs/d455_depth_intrinsics.npy")
rgb0 = cv2.cvtColor(cv2.imread("outputs/d455_color_frame.png"), cv2.COLOR_BGR2RGB)
depth0_m = np.load("outputs/d455_depth_map.npy")
mask0 = np.load("outputs/initial_mask.npy")
mesh_path = "outputs/any6d_model.obj"
mesh_b64 = encode_mesh_to_b64(mesh_path)
payload = {
    "mesh_obj_b64": mesh_b64,        # now this is the actual file
    "K": K.tolist(),
    "rgb": encode_rgb_to_b64(rgb0),
    "depth_npy": encode_npy_to_b64(depth0_m),
    "mask_npy": encode_npy_to_b64(mask0),
}
r = requests.post("http://127.0.0.1:3000/init_register", json=payload, timeout=10)
print(r.json())

# load rgb and depth video
depth_video = np.load("outputs/veo3_video_480p_aligned.npy")  # (T,H,W)
rgb_video_path = "outputs/veo3_video_480p.mp4"
cap = cv2.VideoCapture(rgb_video_path)
rgb_video = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_video.append(rgb_frame)
rgb_video = np.array(rgb_video)  # (T,H,W,3)

# send each frame for tracking
poses = []
for i in range(depth_video.shape[0]):
    rgb = rgb_video[i]
    depth_m = depth_video[i]
    payload = {
        "rgb": encode_rgb_to_b64(rgb),
        "depth_npy": encode_npy_to_b64(depth_m),
    }
    r = requests.post("http://127.0.0.1:3000/track", json=payload, timeout=0.5)
    pose_4x4 = np.array(r.json()["pose"]["pose_4x4"]).reshape(4, 4)
    poses.append(pose_4x4)
    print(f"Frame {i+1}/{depth_video.shape[0]}: pose=\n{pose_4x4}")
poses = np.array(poses)  # (T,4,4)
np.save("outputs/foundationpose_poses.npy", poses)

