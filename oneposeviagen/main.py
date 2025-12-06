import cv2
from cv2.gapi import mask
import numpy as np
import trimesh
import open3d as o3d
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.optimize import minimize
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras,
                                PointLights, RasterizationSettings, BlendParams,
                                MeshRenderer, MeshRasterizer, SoftPhongShader)
from pytorch3d.io import load_objs_as_meshes
from estimater import *
from datareader import *

from oneposeviagen.match_pairs import image_pair_matching


def fit_plane(pcd, filter_z=1, distance_threshold=0.002):
    points = np.asarray(pcd.points)
    mask = points[:, 2] <= filter_z
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=distance_threshold,
                                                       ransac_n=3,
                                                       num_iterations=1000)
    points = np.asarray(pcd.points)
    projection = points / points[:, 2:3]
    projection = projection[:, :2]
    min_x, max_x = np.min(projection[:, 0]), np.max(projection[:, 0])
    min_y, max_y = np.min(projection[:, 1]), np.max(projection[:, 1])
    field_of_view = np.array([min_x, min_y, max_x, max_y])
    return plane_model, field_of_view

def render_image(mesh, camera_poses, width=640, height=480, fov=1, device='cpu'):
    # Handle both numpy arrays and tensors
    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.to(device)
    else:
        camera_poses = torch.tensor(camera_poses, device=device)
    if len(camera_poses.shape) == 2:
        camera_poses = camera_poses[None, :]
    # Render and save images from different camera poses
    mesh = load_objs_as_meshes([mesh], device=device)
    R = camera_poses[:, :3, :3]
    T = camera_poses[:, 3, :3]
    num_poses = camera_poses.shape[0]
    cameras = PerspectiveCameras(R=R, T=T, device=device,
        focal_length=torch.ones(num_poses, 1) * 0.5 * width / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width/2, height/2)).repeat(num_poses).reshape(-1, 2),  #different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False)
    light_location = torch.linalg.inv(camera_poses)[:, 3, :3]
    lights = PointLights(location=light_location, device=device)
    raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0,0,0)),
            cameras=cameras,
            lights=lights
        )
    )
    fragments = renderer.rasterizer(mesh.extend(num_poses))
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)
    return color, depth

def render_multi_images(mesh, width=640, height=480, fov=1, radius=3.0, num_samples=6, num_ups=2, sample_flag = 0, input_pose = np.eye(4), device='cpu'):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    # aspect_ratio = width / height # modified
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    color, depth = render_image(mesh, camera_poses, width, height, fov, device)
    return color, depth, camera_poses, camera_intrinsics

def sample_camera_poses(radius, num_samples, num_up_samples=4, device='cpu'):
    '''
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    '''
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                # import pdb
                # pdb.set_trace()
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up, device="cuda:0")
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R.detach().cpu().numpy()
                camera_pose[3, :3] = T.detach().cpu().numpy()
                camera_poses.append(camera_pose)
    return torch.tensor(np.array(camera_poses), device=device)

def project_2d_to_3d(image_points, depth, camera_intrinsics, camera_pose):
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    # Convert image points to normalized device coordinates (NDC)
    ndc_points = np.zeros((image_points.shape[0], 3))
    for i, (u, v) in enumerate(image_points):
        z = depth[int(v), int(u)]
        x = - (u - cx) * z / fx
        y = - (v - cy) * z / fy
        ndc_points[i] = [x, y, z]
    valid_mask = ndc_points[:, 2] > 0
    ndc_points = ndc_points[valid_mask]
    # ndc_points = np.vstack((ndc_points, np.zeros(3), [[0, 0, 0]])) # modified
    # Convert from camera coordinates to world coordinates
    ndc_points_homogeneous = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1))))
    world_points_homogeneous = ndc_points_homogeneous @ np.linalg.inv(camera_pose)
    return world_points_homogeneous[:, :3], valid_mask

def select_point(pcd, match_points_on_raw, img_size):
    points = np.asarray(pcd.points)
    projection = points / points[:, 2:3]
    projection = projection[:, :2]
    # plt.plot(projection[:, 0], projection[:, 1], 'o')
    # plt.show()

    # bounding box of projection
    min_x, max_x = np.min(projection[:, 0]), np.max(projection[:, 0])
    min_y, max_y = np.min(projection[:, 1]), np.max(projection[:, 1])

    # project to image
    projection[:, 0] = (projection[:, 0] - min_x) / (max_x - min_x) * img_size[1]
    projection[:, 1] = (projection[:, 1] - min_y) / (max_y - min_y) * img_size[0]

    # select points closest to match points
    closest_points = []
    for raw_point in match_points_on_raw:
        distances = np.linalg.norm(projection - raw_point, axis=1)
        closest_index = np.argmin(distances)
        closest_points.append(points[closest_index])

    # show the selected points in 3D
    selected = o3d.geometry.PointCloud()
    selected.points = o3d.utility.Vector3dVector(closest_points)
    # o3d.visualization.draw_geometries([pcd, selected])
    return closest_points

def get_scale(mesh_path,
              mesh,
              cropped_rgb,
              rgb_image, 
              pcd_legacy,
              mask_box, 
              out_dir,
              device='cuda:0'):
    plane_model, field_of_view = fit_plane(pcd_legacy, 1, 0.002)
    fov = np.arctan((field_of_view[2] - field_of_view[0]) / 2) * 2
    bounding_box = mesh.bounds
    center = (bounding_box[0] + bounding_box[1]) / 2
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    fov_radians = fov / 2
    radius = (max_dimension / 2) / np.tan(fov_radians)
    radius = 2 * radius
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] rendering radius', radius)

    # Render multimle images and feature matching
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] rendering objects...')
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(mesh_path, 
                                                                          rgb_image.shape[1],
                                                                          rgb_image.shape[0], fov, radius=radius,
                                                                          num_samples=8, num_ups=1, device=device, sample_flag=0, input_pose=np.eye(4))
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    # Convert cropped_rgb to grayscale to match the working code
    cropped_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2GRAY)
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] matching features...')
    best_pose, match_result = image_pair_matching(grays, cropped_gray, out_dir,
                                    resize=[-1], viz=False, save=False, keypoint_threshold=0.001, match_threshold=0.01)
    chosen_pose = camera_poses[best_pose].cpu().numpy()
    # print('best_pose', np.array2string(chosen_pose, separator=', '))
    # print('matched point number', np.sum(match_result['matches']>-1))
    plt.imsave(os.path.join(out_dir, 'initial_pose.png'), colors[0])
    plt.imsave(os.path.join(out_dir, 'best_pose_rendering.png'), colors[best_pose])
    # chosen_pose[:, 1:3] = -chosen_pose[:, 1:3] # Change due to pyrender's special coordinates

    # Matched points on mesh
    image_points = match_result['keypoints0'][match_result['matches']>-1]
    world_points, valid_mask = project_2d_to_3d(image_points, depths[best_pose],
                                                camera_intrinsics, chosen_pose)
    image_points = image_points[valid_mask]
    # plot_mesh_with_points(mesh, world_points, os.path.join(out_dir, 'points_on_3D.png'))
    # plot_image_with_points(depths[best_pose], image_points, os.path.join(out_dir, 'points_on_2D.png'))

    # Matched points on original picture
    match_points_on_mask = match_result['keypoints1'][match_result['matches'][match_result['matches']>-1]]
    match_points_on_mask = match_points_on_mask[valid_mask]
    sclae_x = (mask_box[3]-mask_box[1]) / cropped_rgb.shape[1]
    sclae_y = (mask_box[2]-mask_box[0]) / cropped_rgb.shape[0]
    match_points_on_raw = match_points_on_mask * np.array([sclae_x, sclae_y]) + np.array([mask_box[1], mask_box[0]])
    # plot_image_with_points(raw_img, match_points_on_raw, os.path.join(out_dir, 'points_original.png'))

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        np.float32(world_points),
        np.float32(match_points_on_raw),
        np.float32(camera_intrinsics),
        distCoeffs=np.zeros(4, dtype=np.float32),
        iterationsCount=100,       # 最大迭代次数
        reprojectionError=8.0,      # 投影误差阈值
        confidence=0.99,            # 置信度
        flags=cv2.SOLVEPNP_ITERATIVE  # 可替换为 SOLVEPNP_EPNP 等
    )
    # assert success
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    world_2_cam = np.eye(4, dtype=np.float32)
    world_2_cam[:3, :3] = rotation_matrix
    world_2_cam[:3, 3] = tvec.squeeze()
    # print('Optimized pose', world_2_cam)

    world_2_cam_render = np.eye(4, dtype=np.float32)
    world_2_cam_render[:3, :3] = np.linalg.inv(rotation_matrix)
    world_2_cam_render[3, :3] = tvec.squeeze() # change due to pytorch3D setting
    world_2_cam_render[:, :2] = -world_2_cam_render[:, :2] # change due to pytorch3D setting
    # print('Optimized pose for rendering', np.array2string(world_2_cam_render, separator=', '))
    color, _ = render_image(mesh_path, world_2_cam_render, rgb_image.shape[1], rgb_image.shape[0], fov, device)
    plt.imsave(os.path.join(out_dir, 'optimized_rendering.png'), color[0])

    # rescaled mesh points
    # mesh points in camera space
    mesh_points = np.hstack((world_points, np.ones((world_points.shape[0], 1)))) @ (world_2_cam).T
    mesh_points = mesh_points[:, :3]
    pcd_points = select_point(pcd_legacy, match_points_on_raw, rgb_image.shape)

    def objective(scale, mesh_points, pcd_points, plane_model):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 0.25
    result = minimize(objective, initial_scale, args=(mesh_points, pcd_points, plane_model), method='L-BFGS-B')
    optimal_scale = result.x[0]
    S = np.array([[optimal_scale, 0, 0, 0],
                  [0, optimal_scale, 0, 0],
                  [0, 0, optimal_scale, 0],
                  [0, 0, 0, 1]])
    M = np.dot(S, world_2_cam)
    return M, optimal_scale

def get_single_pose(color, depth, mask, intrinsic, mesh, topic, num, debug, est_refine_iter=5):
    set_logging_format()
    set_seed(0)
    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = os.path.join(code_dir, "results", topic)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    K = intrinsic
    if len(mask.shape)==3:
        for c in range(3):
            if mask[...,c].sum()>0:
                mask = mask[...,c]
                break
    mask = mask.astype(bool)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
    os.makedirs(f'{debug_dir}/{topic}', exist_ok=True)
    np.savetxt(f'{debug_dir}/{topic}/{num}.txt', pose.reshape(4,4))

    return pose.reshape(4,4)

def load_data(color_img_path: str, 
              depth_npy_path: str,
              mask_img_path: str,
              mesh_path: str,
              intrinsics_path: str,
              out_dir: str = "outputs"):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rgb_image = cv2.imread(color_img_path)
    depth_image = np.load(depth_npy_path)
    mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    mesh = trimesh.load(mesh_path)
    mesh_rotated = mesh.copy()
    intrinsics = np.load(intrinsics_path)
    binary_mask = (mask_image > 0).astype(np.uint8)
    masked_rgb = rgb_image.copy()
    masked_rgb[binary_mask == 0] = 0
    coords = np.argwhere(binary_mask)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    mask_box = [xmin, ymin, xmax, ymax]
    cropped_rgb = rgb_image[ymin:ymax, xmin:xmax]
    # cropped_depth = depth_image[ymin:ymax, xmin:xmax]
    # cropped_mask = mask_image[ymin:ymax, xmin:xmax]
    color_o3d = o3d.t.geometry.Image(rgb_image.astype(np.uint8))
    depth_o3d = o3d.t.geometry.Image(depth_image.astype(np.float32))
    rgbd = o3d.t.geometry.RGBDImage(color_o3d, depth_o3d)
    intrinsic_tensor = o3d.core.Tensor(intrinsics, dtype=o3d.core.Dtype.Float64)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, 
        intrinsic_tensor,
        depth_scale=1.0,
        depth_max=3.0 # Maximum depth to consider (in meters)
    )
    pcd_legacy = pcd.to_legacy()
    


    scales = []
    poses = []

    # get_scale
    M, scale = get_scale(mesh_path,
                         mesh=mesh,
                         cropped_rgb=cropped_rgb,
                         rgb_image=rgb_image,
                         pcd_legacy=pcd_legacy,
                         mask_box=mask_box,
                         out_dir=out_dir,
                         device=device)
    scales.append(scale)
    scale_matrix = np.array([[scale, 0, 0, 0],
                             [0, scale, 0, 0],
                             [0, 0, scale, 0],
                             [0, 0, 0, 1]])
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] initial scale: {scale}")
    mesh.apply_transform(scale_matrix)
    mesh_rotated.apply_transform(scale_matrix)

    for i in range(3):
        #使用Foundationpose进行位姿估计
        pose = get_single_pose(rgb_image, depth_image, mask_image, intrinsics,
                               mesh_rotated, topic="test", num=i, debug=0, est_refine_iter=3)
        poses.append(pose)
        #进行模型旋转
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = pose[0:3, 0:3]
        mesh_rotated.apply_transform(rotation_matrix)
        mesh_rotated.export(os.path.join(out_dir, "rotated_model.obj"))
        rotated_mesh_dir = os.path.join(out_dir, "rotated_model.obj")
        #根据得到的位姿再次进行转换矩阵估计
        scale_output = os.path.join(out_dir, f"iteration_{i+1}")
        os.makedirs(scale_output, exist_ok=True)
        M, scale = get_scale(rotated_mesh_dir,
                         mesh=mesh_rotated,
                         cropped_rgb=cropped_rgb,
                         rgb_image=rgb_image,
                         pcd_legacy=pcd_legacy,
                         mask_box=mask_box,
                         out_dir=scale_output,
                         device=device)
        if scale > 1.5 or scale < 0.5:
           i = i - 1
           continue
        scales.append(scale)
        scale_matrix = np.array([[scale, 0, 0, 0],
                                [0, scale, 0, 0],
                                [0, 0, scale, 0],
                                [0, 0, 0, 1]])
        #进行模型缩放
        mesh.apply_transform(scale_matrix)
        mesh_rotated.apply_transform(scale_matrix)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] final scale: {scales}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] final pose: {poses}")
    final_scale = 1.0
    for scale in scales:
      final_scale *= scale
    mesh.export(out_dir+"/final_model.obj")
    return mesh, pose, final_scale



def main():
    load_data(color_img_path="oneposeviagen/data/color.png",
              depth_npy_path="oneposeviagen/data/depth.npy",
              mask_img_path="oneposeviagen/data/mask.png",
              mesh_path="oneposeviagen/data/sam3d_mustard_tex.obj",
              intrinsics_path="oneposeviagen/data/intrinsics.npy",
              out_dir="oneposeviagen/outputs")

if __name__ == "__main__":
    main()