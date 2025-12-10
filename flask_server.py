#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import argparse
import logging
import base64

import numpy as np
import cv2
import trimesh
from flask import Flask, request, jsonify

from estimater import *  # ScorePredictor, PoseRefinePredictor, FoundationPose, dr.RasterizeCudaContext

app = Flask(__name__)

# ---------------- Global state ----------------

est = None
K_global = None
mesh_global = None
to_origin_global = None
bbox_global = None
initialized = False

EST_REFINE_ITERS = 5
TRACK_REFINE_ITERS = 2

# ---------------- Utils ----------------

def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def build_estimator_from_mesh(mesh: trimesh.Trimesh):
    """Create FoundationPose estimator and bbox info from a trimesh mesh."""
    global mesh_global, to_origin_global, bbox_global

    mesh_global = mesh
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    to_origin_global = to_origin
    bbox_global = bbox

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est_local = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir="./debug_flask",
        debug=1,
        glctx=glctx
    )
    logging.info("Estimator initialization done.")
    return est_local

def load_mesh_from_path(mesh_file: str):
    logging.info(f"Loading mesh from path: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    return mesh

def load_mesh_from_b64(mesh_obj_b64: str):
    """Load a mesh from base64 encoded .obj content."""
    mesh_bytes = base64.b64decode(mesh_obj_b64)
    bio = io.BytesIO(mesh_bytes)
    mesh = trimesh.load(bio, file_type="obj")
    logging.info("Mesh loaded from base64 payload.")
    return mesh

def decode_image_from_base64(img_b64: str):
    """Decode base64 JPEG/PNG -> RGB uint8 np.ndarray (H, W, 3)."""
    img_bytes = base64.b64decode(img_b64)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return rgb

def decode_npy_from_base64(npy_b64: str):
    """Decode base64 .npy bytes -> np.ndarray."""
    raw = base64.b64decode(npy_b64)
    arr = np.load(io.BytesIO(raw))
    return arr

def pose_to_json_dict(T: np.ndarray):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    return {
        "pose_4x4": T.reshape(-1).tolist(),
        "R": R.tolist(),
        "t": t.tolist()
    }

# ---------------- Flask routes ----------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "initialized": initialized,
        "has_estimator": est is not None,
    })

@app.route("/init_register", methods=["POST"])
def init_register():
    """
    Initialize FoundationPose with first frame.

    Expected JSON fields:
    - mesh_obj_b64 (optional): base64 encoded .obj file bytes.
        If present, server will create an estimator from this mesh.
        If not present, server must already have an estimator (from --mesh_file).
    - K: 3x3 list of floats
    - rgb: base64 JPEG/PNG
    - depth_npy: base64 .npy, float32 depth in meters, shape (H, W)
    - mask_npy: base64 .npy, bool or uint8 0/1, shape (H, W)
    """
    global est, K_global, initialized, EST_REFINE_ITERS

    try:
        data = request.get_json(force=True)

        mesh_obj_b64 = data.get("mesh_obj_b64", None)
        if mesh_obj_b64 is not None:
            mesh = load_mesh_from_b64(mesh_obj_b64)
            est_local = build_estimator_from_mesh(mesh)
            # override global estimator
            globals()["est"] = est_local

        if est is None:
            return jsonify({
                "success": False,
                "error": "Estimator not initialized. Provide mesh_obj_b64 in this request or start server with --mesh_file."
            }), 400

        # K
        K_list = data["K"]
        K = np.array(K_list, dtype=np.float64).reshape(3, 3)
        K_global = K

        # Decode rgb, depth, mask
        rgb = decode_image_from_base64(data["rgb"])
        depth_m = decode_npy_from_base64(data["depth_npy"]).astype(np.float32)
        mask_arr = decode_npy_from_base64(data["mask_npy"])

        if mask_arr.dtype != bool:
            mask = mask_arr.astype(bool)
        else:
            mask = mask_arr

        logging.info(f"[init_register] rgb shape={rgb.shape}, depth shape={depth_m.shape}, mask shape={mask.shape}")

        pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth_m,
            ob_mask=mask,
            iteration=EST_REFINE_ITERS
        )

        initialized = True
        pose_dict = pose_to_json_dict(pose)

        return jsonify({
            "success": True,
            "initialized": initialized,
            "pose": pose_dict
        })

    except Exception as e:
        logging.exception("Error in /init_register")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/track", methods=["POST"])
def track():
    """
    Track object in subsequent frames.

    Expected JSON fields:
    - rgb: base64 JPEG/PNG
    - depth_npy: base64 .npy, float32 depth in meters
    """
    global est, K_global, initialized, TRACK_REFINE_ITERS

    if est is None or not initialized or K_global is None:
        return jsonify({"success": False, "error": "Estimator not initialized. Call /init_register first."}), 400

    try:
        data = request.get_json(force=True)
        rgb = decode_image_from_base64(data["rgb"])
        depth_m = decode_npy_from_base64(data["depth_npy"]).astype(np.float32)

        pose = est.track_one(
            rgb=rgb,
            depth=depth_m,
            K=K_global,
            iteration=TRACK_REFINE_ITERS
        )

        pose_dict = pose_to_json_dict(pose)
        return jsonify({
            "success": True,
            "pose": pose_dict
        })

    except Exception as e:
        logging.exception("Error in /track")
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    set_logging_format()
    global est, EST_REFINE_ITERS, TRACK_REFINE_ITERS
    EST_REFINE_ITERS = args.est_refine_iter
    TRACK_REFINE_ITERS = args.track_refine_iter

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
