#!/usr/bin/env python

import argparse
import collections
import datetime
import json

import cv2
import gdown
import imgviz
import IPython
from loguru import logger
import numpy as np
import path
import pybullet_planning as pp
import torch

import mercury
from mercury.examples.picking import _agent
from mercury.examples.picking import _env
from mercury.examples.picking import _get_heightmap
from mercury.examples.picking import _utils

import cv_bridge
from morefusion_ros.msg import ObjectClassArray
from morefusion_ros.msg import ObjectPoseArray
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from _message_subscriber import MessageSubscriber
from base_task_interface import BaseTaskInterface


here = path.Path(__file__).abspath().parent


class SafepickingTaskInterface:
    def __init__(self, base: BaseTaskInterface):
        self.base = base

        self._picking_env = _env.PickFromPileEnv()

        example_dir = path.Path(mercury.__file__).parent / "examples/picking"
        if 1:
            # 0.465
            model = "fusion_net"
            weight_dir = (
                example_dir
                / "logs/20210709_005731-fusion_net-noise/weights/84500"
            )
            gdown.cached_download(
                id="1MBfMHpfOrcMuBFHbKvHiw6SA5f7q1T6l",
                path=weight_dir / "q.pth",
                md5="886b36a99c5a44b54c513ec7fee4ae0d",
            )
        if 0:
            # 0.487
            model = "openloop_pose_net"
            weight_dir = (
                example_dir
                / "logs/20210709_005731-openloop_pose_net-noise/weights/90500"
            )
            gdown.cached_download(
                id="1qtfAKoUiZ2S3AJWJBdhphugJzCpr_Q9g",
                path=weight_dir / "q.pth",
                md5="ab29d8bbfd61d115215c2bad05609279",
            )
        if 0:
            # 0.507
            model = "conv_net"
            weight_dir = (
                example_dir / "logs/20210706_194543-conv_net/weights/91500"
            )
            gdown.cached_download(
                id="1mcI34DQunVDbc5F4ENhkk1-MRspQEleH",
                path=weight_dir / "q.pth",
                md5="ebf2e7b874f322fe7f38d0e39375d943",
            )

        self._agent = _agent.DqnAgent(env=self._picking_env, model=model)
        self._agent.build(training=False)
        self._agent.load_weights(weight_dir)

        self._target_class_id = None
        self._sub_singleview = MessageSubscriber(
            [
                ("/camera/aligned_depth_to_color/image_raw", Image),
                (
                    "/camera/mask_rcnn_instance_segmentation/output/class",
                    ObjectClassArray,
                ),
                (
                    "/camera/mask_rcnn_instance_segmentation/output/label_ins",
                    Image,
                ),
                ("/singleview_3d_pose_estimation/output", ObjectPoseArray),
            ]
        )

    def _get_grasp_poses(self):
        camera_to_base = self.obs["camera_to_base"]
        depth = self.obs["depth"]
        label = self.obs["label"]
        pcd_in_camera = self.obs["pcd_in_camera"]
        pcd_in_base = self.obs["pcd_in_base"]

        normals_in_camera = mercury.geometry.normals_from_pointcloud(
            pcd_in_camera
        )

        instance_id = self.obs["class_id_to_instance_ids"][
            self._target_class_id
        ][0]
        mask = ~np.isnan(depth) & (label == instance_id)

        mask = mask.astype(np.uint8)
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(
            mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )
        contour_mask = cv2.drawContours(
            image=contour_mask,
            contours=contours,
            contourIdx=-1,
            color=1,
            thickness=10,
        )
        contour_mask = contour_mask.astype(bool)
        mask = mask.astype(bool)

        imgviz.io.imsave(
            "/tmp/_get_grasp_poses.jpg",
            imgviz.tile(
                [
                    imgviz.depth2rgb(depth),
                    imgviz.label2rgb(label),
                    imgviz.bool2ubyte(mask),
                    imgviz.bool2ubyte(contour_mask),
                ],
                border=(255, 255, 255),
            ),
        )

        mask = mask & ~contour_mask

        pcd_in_camera = pcd_in_camera[mask]
        pcd_in_base = pcd_in_base[mask]
        normals_in_camera = normals_in_camera[mask]

        normals_in_base = (
            mercury.geometry.transform_points(
                pcd_in_camera + normals_in_camera,
                mercury.geometry.transformation_matrix(*camera_to_base),
            )
            - pcd_in_base
        )
        quaternion_in_base = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normals_in_base
        )

        grasp_poses = np.hstack((pcd_in_base, quaternion_in_base))
        return grasp_poses

    def _look_at_pile(self, *args, **kwargs):
        self.base.look_at(
            eye=[0.5, 0, 0.7], target=[0.5, 0, 0], *args, **kwargs
        )

    def _start_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/request", Empty)
            client.call()

    def _stop_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/stop", Empty)
            client.call()

    def capture_pile(self, wait_for_target=True):
        if self.base._env.object_ids:
            for obj_id in self.base._env.object_ids:
                pp.remove_body(obj_id)
        self.base._env.object_ids = None
        self.base._env.fg_object_id = None

        self._look_at_pile()

        stamp = rospy.Time.now()
        self._sub_singleview.subscribe()
        self._start_passthrough()
        while True:
            if not self._sub_singleview.msgs:
                continue
            depth_msg, cls_msg, lbl_msg, poses_msg = self._sub_singleview.msgs
            if depth_msg.header.stamp < stamp:
                continue
            if wait_for_target:
                class_ids_detected = [c.class_id for c in cls_msg.classes]
                if self._target_class_id not in class_ids_detected:
                    continue
            break
        self._stop_passthrough()
        self._sub_singleview.unsubscribe()

        camera_msg = rospy.wait_for_message(
            "/camera/color/camera_info", CameraInfo
        )
        (
            depth_msg,
            cls_msg,
            label_msg,
            obj_poses_msg,
        ) = self._sub_singleview.msgs

        K = np.array(camera_msg.K).reshape(3, 3)

        bridge = cv_bridge.CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        # -2: unknown, -1: background, 0: instance_0, 1: instance_1, ...
        label = bridge.imgmsg_to_cv2(label_msg).copy()
        label[label == -2] = -1

        class_id_to_instance_ids = collections.defaultdict(list)
        for cls in cls_msg.classes:
            class_id_to_instance_ids[cls.class_id].append(cls.instance_id)
        class_id_to_instance_ids = dict(class_id_to_instance_ids)

        if self._target_class_id in class_id_to_instance_ids:
            target_instance_id = class_id_to_instance_ids[
                self._target_class_id
            ][0]
        else:
            target_instance_id = None

        camera_to_base = self.base.lookup_transform(
            "panda_link0",
            camera_msg.header.frame_id,
            time=camera_msg.header.stamp,
            timeout=rospy.Duration(1),
        )

        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd_in_base = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_base),
        )

        assert self.base._env.object_ids is None
        self.base._env.object_ids = []
        for i, obj_pose_msg in enumerate(obj_poses_msg.poses):
            pose = obj_pose_msg.pose
            instance_id = obj_pose_msg.instance_id
            class_id = obj_pose_msg.class_id
            position = (pose.position.x, pose.position.y, pose.position.z)
            quaternion = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            obj_to_camera = (position, quaternion)
            obj_to_base = pp.multiply(camera_to_base, obj_to_camera)
            visual_file = mercury.datasets.ycb.get_visual_file(class_id)
            obj_id = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                position=obj_to_base[0],
                quaternion=obj_to_base[1],
            )
            if instance_id == target_instance_id:
                self.base._env.fg_object_id = obj_id
            self.base._env.object_ids.append(obj_id)

        self.obs = dict(
            camera_to_base=camera_to_base,
            K=K,
            depth=depth,
            label=label,
            class_id_to_instance_ids=class_id_to_instance_ids,
            pcd_in_camera=pcd_in_camera,
            pcd_in_base=pcd_in_base,
            target_instance_id=target_instance_id,
        )

    def _plan_grasp(self):
        j_init = self.base.pi.getj()

        grasp_poses = self._get_grasp_poses()

        centroid = np.mean(grasp_poses[:, :3], axis=0)
        dist = np.linalg.norm(grasp_poses[:, :3] - centroid, axis=1)
        argsort = np.argsort(dist)

        for grasp_pose in grasp_poses[argsort]:
            for gamma in np.linspace(-np.pi, np.pi, num=6):
                c = mercury.geometry.Coordinate(grasp_pose[:3], grasp_pose[3:])
                c.rotate([0, 0, gamma])

                self.base.pi.setj(j_init)
                j = self.base.pi.solve_ik(c.pose, n_init=10, validate=True)
                if j is None:
                    print("j_grasp is not found")
                    continue
                j_grasp = j

                c.translate([0, 0, -0.1])
                j = self.base.pi.solve_ik(c.pose, validate=True)
                if j is None:
                    print("j_pre_grasp is not found")
                    continue
                j_pre_grasp = j

                self.base.pi.setj(j_pre_grasp)
                obstacles = (
                    self.base._env.bg_objects + self.base._env.object_ids
                )
                if self.base._env.fg_object_id:
                    obstacles.remove(self.base._env.fg_object_id)
                js_grasp = self.base.pi.planj(j_grasp, obstacles=obstacles)
                if js_grasp is None:
                    print("js_grasp is not found")
                    continue

                self.base.pi.setj(j_init)
                js_pre_grasp = self.base.pi.planj(
                    j_pre_grasp, obstacles=self.base._env.bg_objects
                )
                if js_pre_grasp is None:
                    print("js_pre_grasp is not found")
                    continue

                return dict(
                    j_pre_grasp=j_pre_grasp,
                    js_pre_grasp=js_pre_grasp,
                    j_grasp=j_grasp,
                    js_grasp=js_grasp,
                )

    def run(self, target_class_id):
        self._target_class_id = target_class_id

        self.base.init_workspace()
        self.base.reset_pose()

        self.initialize_heightmap_comparison()

        self.capture_pile()

        result_grasp = self._plan_grasp()

        js_extract = self._plan_extraction(j_grasp=result_grasp["j_grasp"])

        result_place = self._plan_placement(j_init=self.base.pi.homej)

        self.base.movejs(result_grasp["js_pre_grasp"], time_scale=5)
        self.base.movejs(result_grasp["js_grasp"], time_scale=15)

        self.base.start_grasp()
        if self.base._env.fg_object_id is not None:
            with pp.WorldSaver():
                self.base.pi.setj(result_grasp["j_grasp"])
                ee_to_world = self.base.pi.get_pose("tipLink")
            obj_to_world = pp.get_pose(self.base._env.fg_object_id)
            obj_to_ee = pp.multiply(pp.invert(ee_to_world), obj_to_world)
            self.base.pi.attachments = [
                pp.Attachment(
                    self.base.pi.robot,
                    self.base.pi.ee,
                    obj_to_ee,
                    self.base._env.fg_object_id,
                )
            ]

        self.base.movejs(js_extract, time_scale=20)
        self.base.reset_pose()

        self.base.movejs(result_place["js_pre_place"], time_scale=5)
        self.base.movejs(result_place["js_place"], time_scale=15)

        self.base.stop_grasp()
        self.base.pi.attachments = []
        rospy.sleep(6)

        self.base.movejs(result_place["js_place"][::-1], time_scale=10)

        self.base.reset_pose()

        self.finalize_heightmap_comparison()

    def _plan_placement(self, j_init):
        bin_position = [0.2, -0.5, 0.1]

        with pp.WorldSaver():
            self.base.pi.setj(j_init)

            c = mercury.geometry.Coordinate(*self.base.pi.get_pose("tipLink"))
            c.position = bin_position

            j = self.base.pi.solve_ik(c.pose, rotation_axis="z")
            assert j is not None
            j_place = j

            js_place = []
            for _ in range(5):
                self.base.pi.setj(j)
                c = mercury.geometry.Coordinate(
                    *self.base.pi.get_pose("tipLink")
                )
                c.translate([0, 0, -0.02], wrt="local")
                j = self.base.pi.solve_ik(c.pose, validate=True)
                assert j is not None
                js_place.append(j)
            js_place = js_place[::-1]

            j_pre_place = js_place[0]
            js_pre_place = [j_pre_place]

        return dict(
            j_pre_place=j_pre_place,
            js_pre_place=js_pre_place,
            j_place=j_place,
            js_place=js_place,
        )

    def _get_heightmap(self, center_xy):
        center = np.array([center_xy[0], center_xy[1], np.nan])
        aabb = np.array(
            [
                center - self._picking_env.HEIGHTMAP_SIZE / 2,
                center + self._picking_env.HEIGHTMAP_SIZE / 2,
            ]
        )
        aabb[0][2] = self.base._env.TABLE_OFFSET - 0.05
        aabb[1][2] = self.base._env.TABLE_OFFSET + 0.5
        heightmap, _, idmap = _get_heightmap.get_heightmap(
            points=self.obs["pcd_in_base"],
            colors=np.zeros(self.obs["pcd_in_base"].shape, dtype=np.uint8),
            ids=self.obs["label"] + 1,  # -1: background -> 0: background
            aabb=aabb,
            pixel_size=self._picking_env.HEIGHTMAP_PIXEL_SIZE,
        )
        idmap -= 1  # 0: background -> -1: background
        return heightmap, idmap

    def _plan_extraction(self, j_grasp):
        world_saver = pp.WorldSaver()

        self.base.pi.setj(j_grasp)
        grasp_pose = self.base.pi.get_pose("tipLink")

        heightmap, idmap = self._get_heightmap(center_xy=grasp_pose[0][:2])
        target_instance_id = self.obs["target_instance_id"]
        maskmap = idmap == target_instance_id

        num_instance = len(self.base._env.object_ids)
        grasp_flags = np.zeros((num_instance,), dtype=np.uint8)
        object_labels = np.zeros(
            (num_instance, len(self._picking_env.CLASS_IDS)), dtype=np.int8
        )
        object_poses = np.zeros((num_instance, 7), dtype=np.float32)
        for i in range(num_instance):
            obj_id = self.base._env.object_ids[i]
            class_id = _utils.get_class_id(obj_id)
            position, quaternion = pp.get_pose(obj_id)
            grasp_flags[i] = obj_id == self.base._env.fg_object_id
            object_label = self._picking_env.CLASS_IDS.index(class_id)
            object_labels[i] = np.eye(len(self._picking_env.CLASS_IDS))[
                object_label
            ]
            object_poses[i] = np.r_[
                position[0] - grasp_pose[0][0],
                position[1] - grasp_pose[0][1],
                position[2] - self.base._env.TABLE_OFFSET,
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
            ]

        ee_poses = np.zeros(
            (self._picking_env.episode_length, 7), dtype=np.float32
        )
        ee_poses = np.r_[
            ee_poses[1:],
            (
                np.hstack(grasp_pose)
                - [
                    grasp_pose[0][0],
                    grasp_pose[0][1],
                    self.base._env.TABLE_OFFSET,
                    0,
                    0,
                    0,
                    0,
                ]
            )[None],
        ]

        observation = dict(
            heightmap=heightmap.astype(np.float32),
            maskmap=maskmap,
            object_labels_init=object_labels,
            object_poses_init=object_poses.astype(np.float32),
            grasp_flags_init=grasp_flags,
            ee_poses=ee_poses.astype(np.float32),
        )
        for key in observation:
            observation[key] = torch.as_tensor(observation[key])[None]

        js = []
        for i in range(self._picking_env.episode_length):
            with torch.no_grad():
                q = self._agent.q(observation)

            q = q[0].cpu().numpy().reshape(-1)
            action_indices = np.argsort(q)[::-1]

            actions, terminals = action_indices // 2, action_indices % 2

            for action, terminal in zip(actions, terminals):
                if i == self._picking_env.episode_length - 1:
                    terminal = 1
                dx, dy, dz, da, db, dg = self._picking_env.actions[action]

                c = mercury.geometry.Coordinate(
                    *self.base.pi.get_pose("tipLink")
                )
                c.translate([dx, dy, dz], wrt="world")
                c.rotate([da, db, dg], wrt="world")

                j = self.base.pi.solve_ik(c.pose)
                if j is not None:
                    break
            js.extend(self.base.pi.get_cartesian_path(j=j))
            self.base.pi.setj(j)

            if terminal == 1:
                break

            ee_poses = np.r_[
                ee_poses[1:],
                (
                    np.hstack(c.pose)
                    - [
                        grasp_pose[0][0],
                        grasp_pose[0][1],
                        self.base._env.TABLE_OFFSET,
                        0,
                        0,
                        0,
                        0,
                    ]
                )[None],
            ].astype(np.float32)
            observation["ee_poses"] = torch.as_tensor(ee_poses)[None]

        world_saver.restore()

        return js

    def initialize_heightmap_comparison(self):
        if self._target_class_id is None:
            raise RuntimeError("self._target_class_id needs to be set")

        self.base.pi.setj(self.base.pi.homej)
        self.capture_pile(wait_for_target=True)
        heightmap, idmap = self._get_heightmap(
            self.base._env.PILE_POSITION[:2]
        )
        target_mask = idmap == self.obs["target_instance_id"]
        target_mask = cv2.dilate(
            target_mask.astype(np.uint8), kernel=np.ones((7, 7))
        ).astype(bool)

        self._target_mask_init = target_mask
        self._heightmap_init = heightmap

    def finalize_heightmap_comparison(self):
        self.capture_pile(wait_for_target=False)
        heightmap, idmap = self._get_heightmap(
            self.base._env.PILE_POSITION[:2]
        )
        if self.obs["target_instance_id"] is not None:
            target_mask = idmap == self.obs["target_instance_id"]
            target_mask = cv2.dilate(
                target_mask.astype(np.uint8), kernel=np.ones((7, 7))
            ).astype(bool)
        else:
            target_mask = np.zeros_like(self._target_mask_init)

        heightmap1, target_mask1 = self._heightmap_init, self._target_mask_init
        heightmap2, target_mask2 = heightmap, target_mask

        heightmap1[target_mask1 | target_mask2] = np.nan
        heightmap2[target_mask1 | target_mask2] = np.nan

        diff = abs(heightmap1 - heightmap2)

        DIFF_THRESHOLD = 0.01
        diff_mask = diff > DIFF_THRESHOLD

        diff_mask_ratio = diff_mask.mean()
        diff_mean = diff[diff_mask].mean()

        logger.info(
            f"Diff mask: {diff_mask_ratio:.1%}, "
            f"Diff mean: {diff_mean:.3f} [m]"
        )

        depth2rgb = imgviz.Depth2RGB()
        viz = imgviz.tile(
            [
                depth2rgb(heightmap1),
                depth2rgb(heightmap2),
                imgviz.bool2ubyte(diff_mask),
            ],
            shape=(1, 3),
            border=(255, 255, 255),
        )

        now = datetime.datetime.now()
        log_dir = here / "logs" / now.strftime("%Y%m%d_%H%M%S.%f")
        log_dir.makedirs_p()
        np.save(log_dir / "heightmap1.npy", heightmap1)
        np.save(log_dir / "heightmap2.npy", heightmap2)
        np.save(log_dir / "diff.npy", diff)
        imgviz.io.imsave(log_dir / "visualization.jpg", viz)
        with open(log_dir / "data.json", "w") as f:
            json.dump(
                dict(
                    timestamp=now.isoformat(),
                    DIFF_THRESHOLD=DIFF_THRESHOLD,
                    diff_mask_ratio=float(diff_mask_ratio),
                    diff_mean=float(diff_mean),
                ),
                f,
                indent=2,
            )
        rospy.loginfo(f"Saved to: {log_dir.realpath()}")

    def test_heightmap_comparison(self):
        self._target_class_id = 5

        self.base.reset_pose()
        self.initialize_heightmap_comparison()

        self.base.reset_pose()
        rospy.loginfo("Move an object and press key to continue")
        mercury.pybullet.pause()
        self.finalize_heightmap_comparison()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rospy.init_node("safepicking_task_interface")
    base = BaseTaskInterface()
    self = SafepickingTaskInterface(base=base)  # NOQA

    if args.cmd:
        exec(args.cmd)

    IPython.embed()


if __name__ == "__main__":
    main()
