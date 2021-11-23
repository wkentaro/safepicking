#!/usr/bin/env python

import queue
import time

import imgviz
from loguru import logger
import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury


class Env:
    def __init__(self, mp4=None, waypoints="none"):
        self._use_visual = False
        self._waypoints = waypoints

        pp.connect(mp4=mp4)
        p.setGravity(0, 0, -9.8)

        p.resetDebugVisualizerCamera(
            cameraYaw=59.4000244140625,
            cameraPitch=-11.26436996459961,
            cameraDistance=1.0,
            cameraTargetPosition=(
                0.14779032766819,
                0.20224905014038086,
                0.2679998278617859,
            ),
        )

        with pp.LockRenderer():
            self.pi = mercury.pybullet.PandaRobotInterface()

            pp.add_data_path()
            plane = pp.load_pybullet("plane.urdf")
            mercury.pybullet.set_pose(plane, ((0, 0, 0.03), (0, 0, 0, 1)))

            self._init_objects()

        for _ in range(240):
            pp.step_simulation()
            for obj in mercury.pybullet.get_body_unique_ids():
                pp.set_velocity(obj, linear=[0, 0, 0], angular=[0, 0, 0])
            time.sleep(1 / 240)

    def _init_objects(self):
        raise NotImplementedError

    def get_place_waypoints(self, place_pose):
        raise NotImplementedError

    def magic_grasp(self, obj, pose_error=None):
        obj_to_world = mercury.pybullet.get_pose(obj)
        target = obj_to_world[0]
        waypoints = self.get_place_waypoints(obj_to_world)
        eye = waypoints[0][0]

        T_camera_to_world = mercury.geometry.look_at(eye, target)

        fovy = np.deg2rad(60)
        height = 240
        width = 240
        # mercury.pybullet.draw_camera(
        #     fovy,
        #     height,
        #     width,
        #     pose=mercury.geometry.pose_from_matrix(T_camera_to_world),
        # )
        rgb, depth, segm = mercury.pybullet.get_camera_image(
            T_camera_to_world, fovy, height, width
        )
        imgviz.io.imsave(
            "/tmp/_magic_grasp.jpg",
            imgviz.tile(
                [
                    rgb,
                    imgviz.depth2rgb(depth),
                    imgviz.label2rgb(segm),
                ],
                border=(255, 255, 255),
            ),
        )
        K = mercury.geometry.opengl_intrinsic_matrix(fovy, height, width)
        points = mercury.geometry.pointcloud_from_depth(
            depth,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        points = mercury.geometry.transform_points(points, T_camera_to_world)
        normals = mercury.geometry.normals_from_pointcloud(points)
        if 0:
            mercury.pybullet.draw_points(points)

        mask = ~np.isnan(depth) & (segm == obj)
        points = points[mask]
        normals = normals[mask]

        centroid = points.mean(axis=0)
        dist = np.linalg.norm(points - centroid, axis=1)
        p = np.argsort(dist)[:10]
        points = points[p]
        normals = normals[p]

        if 0:
            for point, normal in zip(points, normals):
                pp.add_line(point, point + normal * 0.05, width=2)
        quaternions = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals
        )
        if 0:
            for position, quaternion in zip(points, quaternions):
                pp.draw_pose((position, quaternion))

        c = mercury.geometry.Coordinate(points[0], quaternions[0])
        c.translate([0, 0, -0.1])
        pre_grasp_pose = c.pose

        obj_to_world = pp.get_pose(obj)

        j = self.pi.solve_ik(pre_grasp_pose, rotation_axis="z")
        self.pi.setj(j)
        for _ in self.pi.grasp(min_dz=0.09, max_dz=0.11):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        ee_to_world = self.pi.get_pose("tipLink")

        if pose_error is not None:
            obj_to_world = (
                obj_to_world[0] + np.asarray(pose_error[0]),
                obj_to_world[1] + np.asarray(pose_error[1]),
            )

        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)

        self.pi.attachments = [
            pp.Attachment(self.pi.robot, self.pi.ee, obj_to_ee, obj)
        ]

        self.pi.setj(self.pi.homej)
        self.pi.attachments[0].assign()

    def run(self, target_obj, pose_error=None):
        self._target_obj = target_obj

        place_pose = pp.get_pose(target_obj)

        self.magic_grasp(target_obj, pose_error)

        with pp.WorldSaver(), pp.LockRenderer():
            with self.pi.enabling_attachments():
                js = []

                waypoints = self.get_place_waypoints(place_pose)
                pp.draw_pose(waypoints[0], length=0.05, width=2)
                j = self.pi.solve_ik(
                    waypoints[0],
                    move_target=self.pi.robot_model.attachment_link0,
                    n_init=10,
                )
                js.append(j)

                j_start = j
                for waypoint in waypoints[1:]:
                    self.pi.setj(j_start)
                    pp.draw_pose(waypoint, length=0.05, width=2)
                    j_end = self.pi.solve_ik(
                        waypoint,
                        move_target=self.pi.robot_model.attachment_link0,
                    )
                    js.extend(self.pi.get_cartesian_path(j=j_end))
                    j_start = j_end

        p.setJointMotorControlArray(
            bodyIndex=self.pi.robot,
            jointIndices=self.pi.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.pi.homej,
            positionGains=np.ones_like(self.pi.homej),
        )
        [pp.step_simulation() for _ in range(240)]

        for _ in self.pi.movej(js[0], speed=0.005):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        max_forces = queue.deque(maxlen=60)
        for _ in (_ for j in js for _ in self.pi.movej(j, speed=0.002)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

            points = p.getContactPoints(
                bodyA=self.pi.robot, linkIndexA=self.pi.ee
            )
            max_force = max(p[9] for p in points) if points else 0
            max_forces.append(max_force)
            MAX_FORCE_THRESHOLD = 50
            if np.mean(max_forces) > MAX_FORCE_THRESHOLD:
                logger.error(
                    "Stopped the robot because of large force "
                    f">{MAX_FORCE_THRESHOLD}N"
                )
                points = p.getContactPoints(bodyA=self.pi.attachments[0].child)
                mercury.pybullet.draw_points(
                    [
                        p[5]
                        for p in points
                        if (p[2], p[4]) != (self.pi.robot, self.pi.ee)
                    ],
                    colors=[1, 0, 0],
                    size=5,
                )
                break

        self.pi.ungrasp()

        for _ in range(240):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        with pp.LockRenderer():
            js = self.pi.get_cartesian_path(js[1])
        for _ in (_ for j in js for _ in self.pi.movej(j, speed=0.002)):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        for _ in self.pi.movej(self.pi.homej):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        mercury.pybullet.pause()
