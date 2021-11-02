#!/usr/bin/env python

import queue
import time

import imgviz
import numpy as np
import pybullet as p
import pybullet_planning as pp

import mercury


class Env:
    def __init__(self):
        pp.connect()
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

        self.pi = mercury.pybullet.PandaRobotInterface()

        pp.add_data_path()
        plane = pp.load_pybullet("plane.urdf")
        mercury.pybullet.set_pose(plane, ((0, 0, 0.03), (0, 0, 0, 1)))

        self._init_bin()
        [pp.step_simulation() for _ in range(240)]

    def _init_bin(self):
        objects = {}

        # bin
        obj = mercury.pybullet.create_bin(
            X=0.33, Y=0.31, Z=0.11, color=(0.7, 0.7, 0.7, 0.5)
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.5105999999999966,
                    -0.004099999999999998,
                    0.08820000000000094,
                ),
                (0.0, 0.0, -0.003999989333334819, 0.9999920000106667),
            ),
        )
        pp.draw_aabb(pp.get_aabb(obj), color=(0, 0, 0, 1))
        objects["bin"] = obj

        parent = obj

        # cracker_box
        class_id = 2
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(1, 0, 0, 1),
            # mass=mercury.datasets.ycb.masses[class_id],
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    -0.03843769431114197,
                    -0.06841924041509628,
                    -0.020100004971027374,
                ),
                (
                    0.6950822472572327,
                    0.003310413332656026,
                    0.7187591791152954,
                    0.01532843615859747,
                ),
            ),
            parent=parent,
        )
        objects["cracker_box"] = obj

        # sugar_box
        class_id = 3
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(0, 1, 0, 1),
            # mass=mercury.datasets.ycb.masses[class_id],
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    -0.059823607206344604,
                    0.06029653549194336,
                    -0.031200002878904343,
                ),
                (
                    0.6950822472572327,
                    0.003310413332656026,
                    0.7187591791152954,
                    0.01532843615859747,
                ),
            ),
            parent=parent,
        )
        objects["sugar_box"] = obj

        # mustard_bottle
        class_id = 5
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(0, 0, 1, 1),
            # mass=mercury.datasets.ycb.masses[class_id],
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.1017511785030365,
                    -0.06474190950393677,
                    -0.026600003242492676,
                ),
                (
                    0.16755934059619904,
                    0.695159912109375,
                    0.6874131560325623,
                    -0.12704220414161682,
                ),
            ),
            parent=parent,
        )
        objects["mustard_bottle"] = obj

        # tomato_can
        class_id = 4
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=mercury.datasets.ycb.masses[class_id],
            rgba_color=(1, 1, 0, 1),
        )
        pp.set_pose(
            obj,
            (
                (0.5252000000000009, 0.07049999999999995, 0.06700000000000034),
                (
                    -0.03968261331907751,
                    0.7070224261017329,
                    0.7038029876137785,
                    0.05662096621665727,
                ),
            ),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.059504538774490356,
                    0.07194063067436218,
                    -0.02120000123977661,
                ),
                (
                    -0.03331788629293442,
                    0.7073509097099304,
                    0.7032650113105774,
                    0.06295282393693924,
                ),
            ),
            parent=parent,
        )
        objects["tomato_can"] = obj

        for name, obj in objects.items():
            pp.set_dynamics(obj, lateralFriction=0.7)

        self.objects = objects

    def magic_grasp(self, obj):
        obj_to_world = mercury.pybullet.get_pose(obj)
        target = obj_to_world[0]
        c = mercury.geometry.Coordinate(*obj_to_world)
        c.translate([0, 0, 0.2], wrt="world")
        eye = c.position

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
        if 0:
            tiled = imgviz.tile(
                [
                    rgb,
                    imgviz.depth2rgb(depth),
                    imgviz.label2rgb(segm),
                ]
            )
            imgviz.io.pyglet_imshow(tiled)
            imgviz.io.pyglet_run()
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
            time.sleep(1 / 240)

        ee_to_world = self.pi.get_pose("tipLink")
        obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)

        if 1:
            # noise
            position, quaternion = obj_to_ee
            position = obj_to_ee[0] + np.array([0.00, -0.02, 0])
            obj_to_ee = position, quaternion

        self.pi.attachments = [
            pp.Attachment(self.pi.robot, self.pi.ee, obj_to_ee, obj)
        ]

        self.pi.setj(self.pi.homej)
        self.pi.attachments[0].assign()

    def run(self):
        target_obj = self.objects["sugar_box"]
        place_pose = pp.get_pose(target_obj)

        self.magic_grasp(target_obj)

        with pp.WorldSaver():
            with self.pi.enabling_attachments():
                c = mercury.geometry.Coordinate(*place_pose)
                j_goal = self.pi.solve_ik(
                    c.pose,
                    move_target=self.pi.robot_model.attachment_link0,
                    n_init=10,
                )
                c.translate([0, 0, 0.2], wrt="world")
                j_start = self.pi.solve_ik(
                    c.pose,
                    move_target=self.pi.robot_model.attachment_link0,
                    n_init=10,
                )
                self.pi.setj(j_start)
                with pp.LockRenderer():
                    js = self.pi.get_cartesian_path(j=j_goal)

        p.setJointMotorControlArray(
            bodyIndex=self.pi.robot,
            jointIndices=self.pi.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.pi.homej,
            positionGains=np.ones_like(self.pi.homej),
        )
        [pp.step_simulation() for _ in range(240)]

        for _ in self.pi.movej(j_start, speed=0.005):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        max_forces = queue.deque(maxlen=60)
        for _ in (_ for j in js for _ in self.pi.movej(j, speed=0.002)):
            pp.step_simulation()
            self.pi.step_simulation()
            time.sleep(pp.get_time_step())

            points = p.getContactPoints(
                bodyA=self.pi.robot, linkIndexA=self.pi.ee
            )
            max_force = max(p[9] for p in points) if points else 0
            max_forces.append(max_force)
            if np.mean(max_forces) > 50:
                break

        self.pi.ungrasp()

        for _ in range(240):
            pp.step_simulation()
            time.sleep(1 / 240)

        for _ in self.pi.movej(self.pi.homej):
            pp.step_simulation()
            time.sleep(pp.get_time_step())

        mercury.pybullet.pause()


if __name__ == "__main__":
    env = Env()
    env.run()
