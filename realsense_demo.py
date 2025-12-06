#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Real-time FoundationPose with Intel RealSense
# Matches the offline demo's data flow: register once (with ROI+depth mask), then track per-frame.
# Key differences vs your first attempt:
#   - Build K as float64 to match trimesh/default reader dtype
#   - Use cv2.cvtColor for BGR->RGB (avoids negative-stride NumPy views)

import os, time, argparse, logging, cv2, imageio
import numpy as np
import trimesh
import pyrealsense2 as rs

# Your modules
from estimater import *
# Assumes the following utilities are available from your repo:
# depth2xyzmap, toOpen3dCloud, draw_posed_3d_box, draw_xyz_axis

def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

def rs_get_K(intr):
    """Build 3x3 K from pyrealsense2 intrinsics as float64 (fx, fy, ppx, ppy)."""
    # pyrealsense2.intrinsics exposes fx, fy, ppx, ppy. These map to cx, cy as in literature.
    # Keep float64 to match trimesh and your dataset reader's K.
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0,     intr.fy, intr.ppy],
                  [0.0,     0.0,     1.0    ]], dtype=np.float64)
    return K

def rs_to_numpy(frame):
    """Convert RealSense frame to NumPy array."""
    return np.asanyarray(frame.get_data())

def build_mask_from_roi(depth_meters, roi, min_depth=0.05, max_depth=5.0):
    """Very simple mask: first-frame ROI intersected with a sane depth range."""
    x, y, w, h = roi
    H, W = depth_meters.shape
    x2, y2 = min(x+w, W), min(y+h, H)
    mask = np.zeros_like(depth_meters, dtype=bool)
    if w > 0 and h > 0:
        patch = depth_meters[y:y2, x:x2]
        valid = (patch > min_depth) & (patch < max_depth)
        mask[y:y2, x:x2] = valid
    return mask

def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--mesh_file', type=str, default=f'/home/zhengxiao-han/projects/ovo_frito/outputs/any6d_model.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--align_to', type=str, default='color', choices=['color','depth'])
    parser.add_argument('--exposure', type=float, default=None, help='Manual exposure (ms) for RGB; None = auto')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    debug = args.debug
    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)

    # Load mesh and get bbox transform like the offline script
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # Initialize estimator (unchanged)
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est     = FoundationPose(
        model_pts=mesh.vertices,                # trimesh default float64 is fine; we match K to float64
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=debug, glctx=glctx
    )
    logging.info("Estimator initialization done")

    # RealSense setup
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    # Align depth to color (default) or color to depth
    align_to_stream = rs.stream.color if args.align_to == 'color' else rs.stream.depth
    align = rs.align(align_to_stream)

    # Depth scale (meters per unit)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    logging.info(f"[main()] Depth scale: {depth_scale:.9f} m/unit")

    # Optional: manual RGB exposure
    try:
        color_sensor = next(s for s in dev.sensors if s.is_color_sensor())
        if args.exposure is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, float(args.exposure)*1000.0)  # ms -> µs
            logging.info(f"Set manual exposure: {args.exposure} ms")
    except Exception as e:
        logging.warning(f"Could not set exposure: {e}")

    # Intrinsics for the aligned target stream
    if args.align_to == 'color':
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    else:
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    K = rs_get_K(intr)
    logging.info(f"[main()] Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")

    # Warm-up frames for stable exposure/auto-gain
    for _ in range(10):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    # Register once, then track
    pose = None
    frame_id = 0
    t_last = time.time()
    roi = None

    # --- VIDEO RECORDING VARS ---
    recording_frames = []
    recording_start_time = None
    RECORD_DURATION = 10.0  # seconds
    video_saved = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = rs_to_numpy(color_frame)                  # HxWx3 uint8 (BGR)
            depth_raw = rs_to_numpy(depth_frame)                  # HxW uint16
            # Convert to expected formats
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)      # HxWx3 uint8 RGB, contiguous
            rgb = np.ascontiguousarray(rgb)
            depth_m = depth_raw.astype(np.float32) * depth_scale  # meters, float32
            depth_m = np.ascontiguousarray(depth_m)

            if pose is None:
                # Ask ROI once for a crude mask
                if roi is None:
                    tmp = color_bgr.copy()
                    cv2.putText(tmp, "Select ROI around OBJECT, press ENTER",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                    roi = cv2.selectROI("ROI", tmp, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("ROI")

                mask = build_mask_from_roi(depth_m, roi, min_depth=0.05, max_depth=5.0)

                pose = est.register(
                    K=K, rgb=rgb, depth=depth_m, ob_mask=mask,
                    iteration=args.est_refine_iter
                )

                if debug >= 3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth_m, K)
                    valid = depth_m >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
                    o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            else:
                pose = est.track_one(
                    rgb=rgb, depth=depth_m, K=K,
                    iteration=args.track_refine_iter
                )

            # Save pose per frame
            np.savetxt(f'{debug_dir}/ob_in_cam/{frame_id:06d}.txt', pose.reshape(4,4))

            # Visualization (match offline script: draw in RGB, imshow in BGR)
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                vis, ob_in_cam=center_pose, scale=0.1, K=K,
                thickness=3, transparency=0, is_input_rgb=True
            )
            
            # FPS calculation
            fps = 1.0/max(1e-6, time.time()-t_last)
            cv2.putText(vis, f"FPS: {fps:.1f}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            t_last = time.time()

            # --- RECORDING LOGIC ---
            # Only record if tracking (pose is valid) and we haven't saved yet
            if pose is not None and not video_saved:
                if recording_start_time is None:
                    recording_start_time = time.time()
                    logging.info("Tracking started: Recording video...")
                
                # Append current frame (Convert RGB 'vis' back to BGR for VideoWriter)
                recording_frames.append(vis[..., ::-1])

                # Check duration
                elapsed = time.time() - recording_start_time
                remaining = max(0, RECORD_DURATION - elapsed)
                cv2.putText(vis, f"REC: {remaining:.1f}s", (12, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if elapsed >= RECORD_DURATION:
                    logging.info(f"10 seconds reached. Saving {len(recording_frames)} frames...")
                    
                    video_path = os.path.join(debug_dir, "tracking_result.mp4")
                    height, width, _ = recording_frames[0].shape
                    
                    # Initialize VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
                    
                    for f in recording_frames:
                        out.write(f)
                    out.release()
                    
                    logging.info(f"Video saved successfully to: {video_path}")
                    video_saved = True
                    break  # Exit the loop after saving

            if debug >= 1:
                cv2.imshow('FoundationPose Live', vis[..., ::-1])  # RGB -> BGR for OpenCV window
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                elif k == ord('r'):  # re-register
                    pose = None
                    roi = None
                    # Reset recording if re-registering
                    recording_frames = []
                    recording_start_time = None
                    video_saved = False

            if debug >= 2 and not video_saved:
                # Still saving individual frames if debug level is high
                imageio.imwrite(f'{debug_dir}/track_vis/{frame_id:06d}.png', vis)

            frame_id += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
