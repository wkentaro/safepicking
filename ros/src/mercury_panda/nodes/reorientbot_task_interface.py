#!/usr/bin/env python

import argparse
import tempfile

import cv2
import imgviz
import IPython
import numpy as np
import path
import pybullet_planning as pp
import trimesh

import mercury
from mercury.examples.reorientation import _reorient
from mercury.examples.reorientation import _utils

import cv_bridge
from morefusion_panda_ycb_video.msg import ObjectClassArray
from morefusion_panda_ycb_video.msg import ObjectPoseArray
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from _message_subscriber import MessageSubscriber
from _tsdf_from_depth import tsdf_from_depth
from base_task_interface import BaseTaskInterface


class ReorientbotTaskInterface(BaseTaskInterface):
    def __init__(self):
        super().__init__()

        self._subscriber_reorientbot = MessageSubscriber(
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

    def init_task(self):
        # target place
        visual_file = mercury.datasets.ycb.get_visual_file(class_id=3)
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            quaternion=_utils.get_canonical_quaternion(class_id=3),
            rgba_color=(1, 1, 1, 0.5),
            mesh_scale=(0.99, 0.99, 0.99),  # for virtual rendering
        )
        pp.set_pose(
            obj,
            (
                (0.44410000000000166, 0.5560999999999995, 0.02929999999999988),
                (
                    -0.5032839784369476,
                    -0.4819772480647679,
                    -0.4778992452799924,
                    0.5348041517765217,
                ),
            ),
        )
        self._obj_goal = obj
        self._env.PLACE_POSE = pp.get_pose(self._obj_goal)
        c = mercury.geometry.Coordinate(*self._env.PLACE_POSE)
        c.translate([0, 0, 0.2], wrt="world")
        self._env.PRE_PLACE_POSE = c.pose
        # highlight target pose
        mesh = trimesh.load(visual_file)
        mesh.apply_transform(
            mercury.geometry.transformation_matrix(*self._env.PLACE_POSE)
        )
        pp.draw_aabb(mesh.bounds, color=(1, 0, 0, 1))

        # self.scan_pointcloud()
        # annotate_pose(obj)

    def run(self):
        self.init_workspace()
        self.init_task()
        target_class_id = _utils.get_class_id(self._obj_goal)

        self.look_at_pile()

        self._subscriber_reorientbot.subscribe()
        self.start_passthrough()
        while True:
            rospy.sleep(0.1)
            if not self._subscriber_reorientbot.msgs:
                continue
            obj_poses_msg = self._subscriber_reorientbot.msgs[3]
            if target_class_id not in [
                pose.class_id for pose in obj_poses_msg.poses
            ]:
                continue
            break
        self.stop_passthrough()
        self._subscriber_reorientbot.unsubscribe()

        self.rosmsgs_to_env()

        result = self.pick_and_place()
        del result

        IPython.embed()

    def rosmsgs_to_env(self, tsdf=True):
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

        camera_to_base = self.lookup_transform(
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
            self._env.bg_objects.append(bg_structure)

        if self._env.fg_object_id is not None:
            pp.remove_body(self._env.fg_object_id)
        self._env.fg_object_id = obj
        self._env.object_ids = [obj]
        self._env.update_obs()

    def plan_place(self, num_grasps):
        pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(self._env)
        dist_from_centroid = np.linalg.norm(pcd_in_obj, axis=1)

        indices = np.arange(pcd_in_obj.shape[0])
        p = dist_from_centroid.max() - dist_from_centroid

        keep = dist_from_centroid < np.median(dist_from_centroid)
        indices = indices[keep]
        p = p[keep]
        if _utils.get_class_id(self._env.fg_object_id) in [5, 11]:
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

    def pick_and_place(self, num_grasps=10):
        result = self.plan_place(num_grasps=num_grasps)
        if "js_place" not in result:
            rospy.logerr("Failed to plan placement")
            return result

        self.movejs(result["js_pre_grasp"])

        js = self.pi.get_cartesian_path(j=result["j_grasp"])
        self.movejs(js, time_scale=20)

        self.start_grasp()
        rospy.sleep(2)
        self.pi.attachments = result["attachments"]

        self.movejs(result["js_pre_place"], time_scale=5)

        self.movejs(result["js_place"], time_scale=15)

        self.stop_grasp()
        rospy.sleep(9)
        self.pi.attachments = []

        self.movejs(result["js_post_place"], time_scale=10)

        js = self.pi.planj(
            self.pi.homej,
            obstacles=self._env.bg_objects + self._env.object_ids,
        )
        if js is None:
            self.reset_pose(time_scale=5)
        else:
            self.movejs(js, time_scale=5)
        self.wait_interpolation()

        return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rospy.init_node("reorientbot_task_interface")
    self = ReorientbotTaskInterface()  # NOQA

    if args.cmd:
        exec(args.cmd)

    IPython.embed()


if __name__ == "__main__":
    main()
