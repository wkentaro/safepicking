#!/usr/bin/env python

import argparse
import itertools
import tempfile

import cv2
import imgviz
import IPython
import numpy as np
import path
import pybullet_planning as pp

import mercury
from mercury.examples.reorientation import _reorient
from mercury.examples.reorientation import _utils
from mercury.examples.reorientation.pickable_eval import (
    get_goal_oriented_reorient_poses,  # NOQA
)
from mercury.examples.reorientation.reorient_dynamic import (
    plan_dynamic_reorient,  # NOQA
)

import cv_bridge
from morefusion_ros.msg import ObjectClassArray
from morefusion_ros.msg import ObjectPoseArray
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from _message_subscriber import MessageSubscriber
from _tsdf_from_depth import tsdf_from_depth
from base_task_interface import BaseTaskInterface


class ReorientbotTaskInterface:
    def __init__(self, base: BaseTaskInterface):
        self.base = base

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

    def run_singleview(self):
        self.base.init_workspace()
        self.base.init_task()

        if self.base._env.fg_object_id is None:
            self.capture_pile_singleview()
        else:
            self.captue_target_singleview()

        while True:
            result = self.base.pick_and_place()
            if "js_place" in result:
                break

            self.pick_and_reorient()
            self.scan_target()

    def capture_pile_singleview(self):
        self._look_at_pile()
        self._wait_for_message_singleview()
        self._process_message_singleview()

    def _look_at_pile(self, *args, **kwargs):
        self.base.look_at(
            eye=[0.5, 0, 0.7], target=[0.5, 0, 0], *args, **kwargs
        )

    def _wait_for_message_singleview(self):
        target_class_id = _utils.get_class_id(self._obj_goal)

        stamp = rospy.Time.now()
        self._subscriber_reorientbot.msgs = None
        self._subscriber_reorientbot.subscribe()
        self.base.start_passthrough()
        while True:
            rospy.sleep(0.1)
            if not self._subscriber_reorientbot.msgs:
                continue
            obj_poses_msg = self._subscriber_reorientbot.msgs[3]
            if obj_poses_msg.header.stamp < stamp:
                continue
            if target_class_id not in [
                pose.class_id for pose in obj_poses_msg.poses
            ]:
                continue
            break
        self.base.stop_passthrough()
        self._subscriber_reorientbot.unsubscribe()

    def _process_message_singleview(self, tsdf=True):
        cam_msg = rospy.wait_for_message(
            "/camera/color/camera_info", CameraInfo
        )
        (
            depth_msg,
            cls_msg,
            label_msg,
            obj_poses_msg,
        ) = self._subscriber_reorientbot.msgs

        K = np.array(cam_msg.K).reshape(3, 3)

        bridge = cv_bridge.CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan
        label = bridge.imgmsg_to_cv2(label_msg)

        camera_to_base = self.base.lookup_transform(
            "panda_link0",
            obj_poses_msg.header.frame_id,
            time=obj_poses_msg.header.stamp,
        )

        for obj_pose_msg in obj_poses_msg.poses:
            class_id = obj_pose_msg.class_id
            if class_id != _utils.get_class_id(self._obj_goal):
                continue

            pose = obj_pose_msg.pose
            obj_to_camera = (
                (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ),
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
            )
            obj_to_base = pp.multiply(camera_to_base, obj_to_camera)

            visual_file = mercury.datasets.ycb.get_visual_file(class_id)
            collision_file = mercury.pybullet.get_collision_file(visual_file)
            obj = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                position=obj_to_base[0],
                quaternion=obj_to_base[1],
            )
            break
        else:
            raise RuntimeError("Target object is not found")

        if tsdf:
            target_instance_id = obj_pose_msg.instance_id

            mask = label == target_instance_id
            mask = (
                cv2.dilate(
                    imgviz.bool2ubyte(mask),
                    kernel=np.ones((8, 8)),
                    iterations=3,
                )
                == 255
            )
            depth_masked = depth.copy()
            depth_masked[mask] = np.nan
            tsdf = tsdf_from_depth(depth_masked, camera_to_base, K)
            with tempfile.TemporaryDirectory() as tmp_dir:
                visual_file = path.Path(tmp_dir) / "tsdf.obj"
                tsdf.export(visual_file)
                collision_file = mercury.pybullet.get_collision_file(
                    visual_file, resolution=10000
                )
                bg_structure = mercury.pybullet.create_mesh_body(
                    visual_file=visual_file,
                    collision_file=collision_file,
                    rgba_color=(0.5, 0.5, 0.5, 1),
                )
            self.base._env.bg_objects.append(bg_structure)

        if self.base._env.fg_object_id is not None:
            pp.remove_body(self.base._env.fg_object_id)
        self.base._env.fg_object_id = obj
        self.base._env.object_ids = [obj]
        self.base._env.update_obs()

    def capture_target_singleview(self):
        self._look_at_pile()
        self._wait_for_message_singleview()
        self._process_message_singleview()

    def _look_at_target(self):
        if self.base._env.fg_object_id is None:
            # default
            target = [0.2, -0.5, 0.1]
        else:
            target = pp.get_pose(self.base._env.fg_object_id)[0]
        self.base.look_at(
            eye=[target[0] - 0.1, target[1], target[2] + 0.5],
            target=target,
            rotation_axis="z",
            time_scale=4,
        )

    def pick_and_place(self, num_grasps=10):
        result = self._plan_place(num_grasps=num_grasps)
        if "js_place" not in result:
            rospy.logerr("Failed to plan placement")
            return result

        self.base.movejs(result["js_pre_grasp"], time_scale=3)

        js = self.base.pi.get_cartesian_path(j=result["j_grasp"])
        self.base.movejs(js, time_scale=10)

        self.base.start_grasp()
        rospy.sleep(2)
        self.base.pi.attachments = result["attachments"]

        self.base.movejs(result["js_pre_place"], time_scale=5)

        self.base.movejs(result["js_place"], time_scale=7.5)

        self.base.stop_grasp()
        rospy.sleep(9)
        self.base.pi.attachments = []

        self.base.movejs(result["js_post_place"], time_scale=7.5, retry=True)

        js = self.base.pi.planj(
            self.base.pi.homej,
            obstacles=self.base._env.bg_objects + self.base._env.object_ids,
        )
        if js is None:
            self.base.reset_pose(time_scale=3)
        else:
            self.base.movejs(js, time_scale=3)

        return result

    def _plan_place(self, num_grasps):
        pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(self._env)
        dist_from_centroid = np.linalg.norm(pcd_in_obj, axis=1)

        indices = np.arange(pcd_in_obj.shape[0])
        p = dist_from_centroid.max() - dist_from_centroid

        keep = dist_from_centroid < np.median(dist_from_centroid)
        indices = indices[keep]
        p = p[keep]
        if _utils.get_class_id(self.base._env.fg_object_id) in [5, 11]:
            indices = np.r_[
                np.random.choice(indices, num_grasps, p=p / p.sum()),
            ]
        else:
            indices = np.r_[
                np.random.choice(indices, num_grasps // 2, p=p / p.sum()),
                np.random.permutation(pcd_in_obj.shape[0])[: num_grasps // 2],
            ]

        pcd_in_obj = pcd_in_obj[indices]
        normals_in_obj = normals_in_obj[indices]
        quaternion_in_obj = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, -1], normals_in_obj
        )
        grasp_poses = np.hstack([pcd_in_obj, quaternion_in_obj])  # in obj

        if 0:
            for grasp_pose in grasp_poses:
                pp.draw_pose(
                    np.hsplit(grasp_pose, [3]),
                    parent=self._obj_goal,
                    width=2,
                    length=0.05,
                )

        result = _reorient.plan_place(self._env, grasp_poses)

        return result

    def pick_and_reorient(self, heuristic=False):
        result = self._plan_reorient(heuristic=heuristic)
        if "js_place" not in result:
            rospy.logerr("Failed to plan reorientation")
            return

        self.base.movejs(result["js_pre_grasp"], time_scale=3)

        js = self.base.pi.get_cartesian_path(j=result["j_grasp"])

        self.base.movejs(js, time_scale=10)
        self.base.start_grasp()
        rospy.sleep(2)
        self.base.pi.attachments = result["attachments"]

        js = result["js_place"]
        self.base.movejs(js, time_scale=5)

        with pp.WorldSaver():
            self.base.pi.setj(js[-1])
            c = mercury.geometry.Coordinate(*self.base.pi.get_pose("tipLink"))
            js = []
            for i in range(3):
                c.translate([0, 0, -0.01], wrt="world")
                j = self.base.pi.solve_ik(c.pose, rotation_axis=None)
                if j is not None:
                    js.append(j)
        self.base.movejs(js, time_scale=7.5, wait=False)

        self.base.stop_grasp()

        rospy.sleep(6)
        self.base.pi.attachments = []

        js = result["js_post_place"]
        self.base.movejs(js, time_scale=5)

        self.base.movejs([self.base.pi.homej], time_scale=3)

    def _plan_reorient(self, heuristic=False):
        if heuristic:
            grasp_poses = _reorient.get_grasp_poses(self._env)
            grasp_poses = list(itertools.islice(grasp_poses, 12))
            reorient_poses = _reorient.get_static_reorient_poses(self._env)

            result = {}
            for grasp_pose, reorient_pose in itertools.product(
                grasp_poses, reorient_poses
            ):
                result = _reorient.plan_reorient(
                    self._env, grasp_pose, reorient_pose
                )
                if "js_place" in result:
                    break
            else:
                rospy.logerr("No solution found")
        else:
            (
                reorient_poses,
                pickable,
                target_grasp_poses,
            ) = get_goal_oriented_reorient_poses(self._env)

            grasp_poses = _reorient.get_grasp_poses(self._env)  # in world
            grasp_poses = list(itertools.islice(grasp_poses, 25))

            for threshold in np.linspace(0.99, 0.5):
                indices = np.where(pickable > threshold)[0]
                if indices.size > 100:
                    break
            indices = np.random.choice(
                indices, min(indices.size, 1000), replace=False
            )
            reorient_poses = reorient_poses[indices]
            pickable = pickable[indices]

            result = plan_dynamic_reorient(
                self._env,
                grasp_poses,
                reorient_poses,
                pickable,
            )
        return result

    def capture_pile_multiview(self):
        raise NotImplementedError

    def init_task(self):
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rospy.init_node("reorientbot_task_interface")
    base = BaseTaskInterface()
    self = ReorientbotTaskInterface(base)  # NOQA

    if args.cmd:
        exec(args.cmd)

    IPython.embed()


if __name__ == "__main__":
    main()
