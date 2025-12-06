#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline FoundationPose on (RGB .mp4 + aligned depth .npy) with tracking video save.
- Register once with ROI + depth mask on frame 0
- Track per-frame afterwards
- Save visualization frames to a video file (e.g., MP4 via mp4v)
- Optionally save the 3D trajectory of the object's center.
- Optionally save the full 6D pose trajectory of the object.
"""

import os, argparse, time, logging, cv2, imageio
import numpy as np
import trimesh

# --- Your modules (from your previous script) ---
from estimater import * # FoundationPose, ScorePredictor, PoseRefinePredictor, dr
# from utils import depth2xyzmap, toOpen3dCloud, draw_posed_3d_box, draw_xyz_axis

def set_logging_format():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

def load_K_from_txt(path):
    if path.endswith(".npy"):
        K = np.load(path).astype(np.float64)
        # --- Added Validation ---
        if K.shape != (3, 3):
            raise ValueError(
                f"Loaded K from {path} has shape {K.shape}, "
                f"but expected (3, 3)."
            )
        # --- End Validation ---
    else:
        vals = np.loadtxt(path).reshape(-1)
        if vals.size == 9:
            K = vals.reshape(3,3).astype(np.float64)
        elif vals.size >= 4:
            fx, fy, cx, cy = vals[:4]
            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1. ]], dtype=np.float64)
        else:
            raise ValueError("Unrecognized K format. Provide 3x3 or (fx,fy,cx,cy).")
    return K

def build_mask_from_roi(depth_meters, roi, min_depth=0.05, max_depth=5.0):
    x, y, w, h = roi
    H, W = depth_meters.shape
    x2, y2 = min(x+w, W), min(y+h, H)
    mask = np.zeros_like(depth_meters, dtype=bool)
    if w > 0 and h > 0:
        patch = depth_meters[y:y2, x:x2]
        valid = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)
        mask[y:y2, x:x2] = valid
    return mask

# --- NEW: make safe BGR frames for OpenCV drawing/writing ---
def cv_bgr(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to a C-contiguous BGR uint8 array (no negative strides).
    Avoids 'Layout incompatible with cv::Mat' errors when calling cv2.putText, etc.
    """
    bgr = np.ascontiguousarray(img_rgb_uint8[..., ::-1])  # RGB->BGR and make contiguous
    if not bgr.flags.writeable:
        bgr = bgr.copy()
    if bgr.dtype != np.uint8:
        bgr = bgr.astype(np.uint8, copy=False)
    return bgr

# --- NEW: Get ROI from a mask file ---
def get_roi_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Calculates the bounding box (x,y,w,h) from a 2D boolean/binary mask."""
    if not mask.any():  # Handle empty mask
        logging.warning("[mask] get_roi_from_mask received an empty mask.")
        return (0, 0, 0, 0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Check if there are any True values at all
    if not rows.any() or not cols.any():
        logging.warning("[mask] get_roi_from_mask found no True pixels.")
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    x = int(x_min)
    y = int(y_min)
    w = int(x_max - x_min) + 1 # +1 to be inclusive
    h = int(y_max - y_min) + 1 # +1 to be inclusive

    return (x, y, w, h)
# --- END NEW ---

def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/tsr_mustard0/mesh.obj')
    parser.add_argument('--rgb_video', type=str, required=True, help='Path to RGB .mp4')
    parser.add_argument('--depth_video_npy', type=str, required=True, help='Path to aligned depth video .npy (T,H,W) in meters')
    parser.add_argument('--K_txt', type=str, required=True, help='3x3 K .txt/.npy or fx fy cx cy .txt')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Path to a .npy file (H,W) containing the object mask for frame 0.')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--min_depth', type=float, default=0.05)
    parser.add_argument('--max_depth', type=float, default=5.0)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_offline')
    parser.add_argument('--vis', action='store_true', help='Show live visualization window')

    # ---- video saving options ----
    parser.add_argument('--save_video', action='store_true', help='Save tracking visualization to a video file')
    parser.add_argument('--out_video', type=str, default='tracking.mp4', help='Output video path (e.g., tracking.mp4)')
    parser.add_argument('--fourcc', type=str, default='mp4v', help="FOURCC codec, e.g., 'mp4v', 'avc1'")
    parser.add_argument('--fps_out', type=float, default=30.0, help='Fallback FPS if input FPS is unavailable')

    # ---- trajectory saving options ----
    parser.add_argument('--save_trajectory', action='store_true', help='Save the 3D trajectory of the object center to a .npy file')
    parser.add_argument('--out_trajectory', type=str, default='bbox_trajectory.npy', help='Output 3D trajectory path (e.g., bbox_trajectory.npy)')
    parser.add_argument('--save_pose_trajectory', action='store_true', help='Save the full 6D pose trajectory to a .npy file')
    parser.add_argument('--out_pose_trajectory', type=str, default='pose_trajectory.npy', help='Output 6D pose trajectory path (e.g., pose_trajectory.npy)')

    args = parser.parse_args()

    set_logging_format()
    os.makedirs(args.debug_dir, exist_ok=True)
    os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
    os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)

    # --- Load mesh & set up FoundationPose ---
    mesh = trimesh.load(args.mesh_file)
    # mesh.apply_scale(1.0/10.0)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est     = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer, refiner=refiner,
        debug_dir=args.debug_dir, debug=args.debug, glctx=glctx
    )
    logging.info("[init] Estimator ready. (FoundationPose)")

    # --- Load intrinsics
    K = load_K_from_txt(args.K_txt)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    logging.info(f"[K] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # --- Load depth video (T,H,W) in meters
    depth_video = np.load(args.depth_video_npy)
    if depth_video.ndim != 3:
        raise ValueError("depth_video_npy must be a (T,H,W) array.")
    depth_video = depth_video.astype(np.float32, copy=False)
    T_d, H_d, W_d = depth_video.shape
    logging.info(f"[depth] depth_video: (T,H,W)=({T_d},{H_d},{W_d}), dtype={depth_video.dtype}")

    # --- Open RGB video
    cap = cv2.VideoCapture(args.rgb_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.rgb_video}")
    ok, frame_bgr0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame from RGB video.")
    H_rgb, W_rgb = frame_bgr0.shape[:2]
    logging.info(f"[rgb] video size: (H,W)=({H_rgb},{W_rgb})")

    # --- Ensure depth matches RGB size; if not, resize depth (nearest)
    if (H_d != H_rgb) or (W_d != W_rgb):
        logging.info(f"[resize] resizing depth frames from {(H_d,W_d)} -> {(H_rgb,W_rgb)}")
        depth_rs = np.empty((T_d, H_rgb, W_rgb), dtype=np.float32)
        for i in range(T_d):
            depth_rs[i] = cv2.resize(np.ascontiguousarray(depth_video[i]),
                                     (W_rgb, H_rgb),
                                     interpolation=cv2.INTER_NEAREST)
        depth_video = depth_rs
        T_d, H_d, W_d = depth_video.shape

    # --- Determine FPS for output video
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = input_fps if input_fps and input_fps > 0 else args.fps_out
    logging.info(f"[fps] input_fps={input_fps:.3f} -> out_fps={out_fps:.3f}")

    # --- Optional: set up VideoWriter for saving tracking visualization ---
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)  # e.g., 'mp4v' or 'avc1'
        writer = cv2.VideoWriter(args.out_video, fourcc, out_fps, (W_rgb, H_rgb))
        if not writer.isOpened():
            logging.warning(f"[video] Failed to open writer with FOURCC={args.fourcc}. "
                            f"Try another codec (e.g., mp4v/avc1) or check your OpenCV build.")
        else:
            logging.info(f"[video] Writing to {args.out_video} at {out_fps:.2f} FPS, size=({W_rgb},{H_rgb}), FOURCC={args.fourcc}")

    # --- Initialize trajectory lists ---
    bbox_center_trajectory = []
    pose_trajectory = []

    # --- Frame 0: register
    rgb0 = cv2.cvtColor(frame_bgr0, cv2.COLOR_BGR2RGB)         # uint8 RGB
    rgb0 = np.ascontiguousarray(rgb0)
    depth0 = depth_video[0].astype(np.float32, copy=False)

    # --- MODIFIED: Get ROI from mask file or manually ---
    if args.mask_file:
        logging.info(f"[mask] Loading mask from {args.mask_file}")
        mask0 = np.load(args.mask_file)

        # Check and resize mask if its shape doesn't match the frame
        if mask0.shape != (H_rgb, W_rgb):
            logging.warning(f"[mask] Mask shape {mask0.shape} != frame shape {(H_rgb, W_rgb)}. Resizing mask...")
            mask0 = cv2.resize(
                mask0.astype(np.uint8),
                (W_rgb, H_rgb),
                interpolation=cv2.INTER_NEAREST
            )

        mask0 = mask0.astype(bool)
        roi = get_roi_from_mask(mask0)
        logging.info(f"[mask] Computed ROI: {roi}")

    else:
        # Manual ROI selection
        logging.info("[ROI] Please select the region of interest for the object...")
        tmp = frame_bgr0.copy()
        cv2.putText(tmp, "Select ROI around OBJECT, press ENTER",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        roi = cv2.selectROI("ROI", tmp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI")
        logging.info(f"[ROI] Selected ROI: {roi}")

    ob_mask = build_mask_from_roi(depth0, roi, min_depth=args.min_depth, max_depth=args.max_depth)
    # --- END MODIFICATION ---

    if not ob_mask.any():
        raise RuntimeError("Object mask is empty. Cannot register. "
                           "Check ROI selection or mask file and depth range.")

    pose = est.register(K=K, rgb=rgb0, depth=depth0, ob_mask=ob_mask, iteration=args.est_refine_iter)
    np.savetxt(f'{args.debug_dir}/ob_in_cam/000000.txt', pose.reshape(4,4))

    # --- Record trajectories for frame 0 ---
    if args.save_trajectory:
        center_pose_0 = pose @ np.linalg.inv(to_origin)
        bbox_center_trajectory.append(center_pose_0[0:3, 3])
    if args.save_pose_trajectory:
        pose_trajectory.append(pose)

    def visualize(rgb_uint8, pose_44):
        vis = rgb_uint8.copy()
        try:
            center_pose = pose_44 @ np.linalg.inv(to_origin)
            if 'draw_posed_3d_box' in globals():
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            if 'draw_xyz_axis' in globals():
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                                    thickness=3, transparency=0, is_input_rgb=True)
        except Exception as e:
            cv2.putText(vis, f"viz err: {e}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        return vis

    t_last = time.time()
    frame_id = 1

    # First-frame vis
    vis0_rgb = visualize(rgb0, pose)       # RGB
    vis0_bgr = cv_bgr(vis0_rgb)            # safe BGR
    if writer is not None and writer.isOpened():
        writer.write(vis0_bgr)
    if args.vis or args.debug >= 1:
        cv2.imshow("FoundationPose Offline", vis0_bgr)
        cv2.waitKey(1)

    # --- Main loop: track subsequent frames
    # Stop at min(depth_frames, rgb_frames) if container reports frame count.
    rgb_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else T_d
    limit = min(T_d, rgb_count if rgb_count > 0 else T_d)

    while True:
        if frame_id >= limit:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        depth_m = depth_video[frame_id].astype(np.float32, copy=False)

        pose = est.track_one(rgb=rgb, depth=depth_m, K=K, iteration=args.track_refine_iter)
        np.savetxt(f'{args.debug_dir}/ob_in_cam/{frame_id:06d}.txt', pose.reshape(4,4))

        # --- Record trajectories for current frame ---
        if args.save_trajectory:
            center_pose_current = pose @ np.linalg.inv(to_origin)
            bbox_center_trajectory.append(center_pose_current[0:3, 3])
        if args.save_pose_trajectory:
            pose_trajectory.append(pose)

        vis_rgb = visualize(rgb, pose)
        vis_bgr = cv_bgr(vis_rgb)

        # FPS overlay (after conversion to a safe BGR buffer)
        fps = 1.0 / max(1e-6, time.time() - t_last)
        cv2.putText(vis_bgr, f"FPS: {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
        t_last = time.time()

        # Write to video
        if writer is not None and writer.isOpened():
            writer.write(vis_bgr)

        # Optional display
        if args.vis or args.debug >= 1:
            cv2.imshow("FoundationPose Offline", vis_bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:   # ESC
                break
            elif k == ord('r'): # re-register on this frame
                tmp = frame_bgr.copy()
                cv2.putText(tmp, "Select ROI around OBJECT, press ENTER",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                roi = cv2.selectROI("ROI", tmp, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("ROI")
                ob_mask = build_mask_from_roi(depth_m, roi,
                                              min_depth=args.min_depth, max_depth=args.max_depth)
                pose = est.register(K=K, rgb=rgb, depth=depth_m, ob_mask=ob_mask,
                                    iteration=args.est_refine_iter)
                np.savetxt(f'{args.debug_dir}/ob_in_cam/{frame_id:06d}.txt', pose.reshape(4,4))
                
                # --- Update trajectory if re-registered ---
                # (Overwrite the last recorded pose/center)
                if args.save_trajectory and bbox_center_trajectory:
                    center_pose_new = pose @ np.linalg.inv(to_origin)
                    bbox_center_trajectory[-1] = center_pose_new[0:3, 3]
                if args.save_pose_trajectory and pose_trajectory:
                    pose_trajectory[-1] = pose


        frame_id += 1

    # Cleanup
    if writer is not None:
        writer.release()
        logging.info(f"[video] Saved tracking visualization to {args.out_video}")
    cap.release()
    cv2.destroyAllWindows()

    # --- Save trajectories ---
    if args.save_trajectory:
        traj_array = np.array(bbox_center_trajectory)
        np.save(args.out_trajectory, traj_array)
        logging.info(f"[trajectory] Saved 3D center trajectory ({traj_array.shape}) to {args.out_trajectory}")
        
    if args.save_pose_trajectory:
        pose_traj_array = np.array(pose_trajectory)
        np.save(args.out_pose_trajectory, pose_traj_array)
        logging.info(f"[trajectory] Saved 6D pose trajectory ({pose_traj_array.shape}) to {args.out_pose_trajectory}")
        
    logging.info("[done] Offline tracking complete.")

if __name__ == "__main__":
    main()


"""
python inference_rgbd_demo.py\
    --rgb_video /home/zhengxiao-han/projects/msr_final_project/outputs/test_run/veo3_video_480p.mp4\
    --depth_video_npy /home/zhengxiao-han/projects/msr_final_project/outputs/test_run/veo3_video_480p_aligned.npy\
    --K_txt /home/zhengxiao-han/projects/msr_final_project/outputs/test_run/color_intrinsics.npy\
    --mask_file /home/zhengxiao-han/projects/msr_final_project/outputs/test_run/predicted_mask.npy\
    --vis\
    --save_video\
    --save_trajectory\
    --save_pose_trajectory
"""


