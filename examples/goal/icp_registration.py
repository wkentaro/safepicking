#!/usr/bin/env python

import numpy as np

import mercury
import open3d
import trimesh


def icp_registration(pcd, pcd_v, transform):
    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(pcd)

    target = open3d.geometry.PointCloud()
    target.points = open3d.utility.Vector3dVector(pcd_v)

    threshold = 0.02
    registration = open3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=threshold,
        init=transform,
    )
    return registration.transformation


def draw_registration_result(pcd, pcd_v, T_obj_v_to_world, transform):
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.PointCloud(pcd, [0, 1.0, 0]),
        transform=transform,
    )
    scene.add_geometry(trimesh.PointCloud(pcd_v, [0, 0, 1.0]))
    visual_file = mercury.datasets.ycb.get_visual_file(class_id=2)
    cad = trimesh.load_mesh(visual_file)
    cad.visual = cad.visual.to_color()
    scene.add_geometry(
        cad,
        transform=T_obj_v_to_world,
    )
    scene.show()


def main():
    data = dict(np.load("assets/pcd_001.npz"))

    K = data["K"]
    mask = data["mask"]
    depth = data["depth"]
    aabb_min = data["aabb_min"]
    aabb_max = data["aabb_max"]
    T_camera_to_world = data["T_camera_to_world"]
    T_obj_v_to_world = (
        mercury.geometry.translation_matrix([0, -2, 0])
        @ data["T_obj_v_to_world"]
    )

    pcd = mercury.geometry.pointcloud_from_depth(
        depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )
    pcd = mercury.geometry.transform_points(pcd, T_camera_to_world)
    mask = (
        mask
        & (pcd >= aabb_min).all(axis=2)
        & (pcd <= aabb_max).all(axis=2)
        & ~np.isnan(depth)
    )
    pcd = pcd[mask]

    pcd_v = np.loadtxt(mercury.datasets.ycb.get_pcd_file(class_id=2))
    pcd_v = mercury.geometry.transform_points(pcd_v, T_obj_v_to_world)

    transform = np.eye(4)
    draw_registration_result(pcd, pcd_v, T_obj_v_to_world, transform)
    transform = icp_registration(pcd, pcd_v, transform)
    draw_registration_result(pcd, pcd_v, T_obj_v_to_world, transform)


if __name__ == "__main__":
    main()
