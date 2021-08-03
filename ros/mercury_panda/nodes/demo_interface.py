#!/usr/bin/env python

import enum
import sys
import tempfile
import warnings

import imgviz
import IPython
import numpy as np
import scipy.ndimage
import skimage.draw
import skrobot
import torch

import mercury

import actionlib
from actionlib_msgs.msg import GoalStatus
import cv_bridge
from franka_control.msg import ErrorRecoveryAction
from franka_control.msg import ErrorRecoveryGoal
from geometry_msgs.msg import Pose
import message_filters
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
from std_srvs.srv import SetBoolRequest
import tf

from mercury_panda.msg import ObjectClassArray
from morefusion_panda_ycb_video.msg import ObjectClass as MFObjectClass
from morefusion_panda_ycb_video.msg import (
    ObjectClassArray as MFObjectClassArray,  # NOQA
)
from morefusion_panda_ycb_video.msg import ObjectPoseArray

sys.path.insert(0, "../../../examples/target_pick")

import _agent  # NOQA
import _env  # NOQA
import _get_heightmap  # NOQA


class Panda(skrobot.models.Panda):
    def __init__(self, *args, **kwargs):
        urdf = rospy.get_param("/robot_description")
        tmp_file = tempfile.mktemp()
        with open(tmp_file, "w") as f:
            f.write(urdf)
        super().__init__(urdf_file=tmp_file)

    @property
    def rarm(self):
        link_names = ["panda_link{}".format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [link.joint for link in links]
        model = skrobot.model.RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.panda_suction_cup
        return model


def skrobot_coords_from_matrix(matrix):
    return skrobot.coordinates.Coordinates(
        pos=matrix[:3, 3],
        rot=matrix[:3, :3],
    )


def ros_pose_msg_from_coord(coord):
    pose_msg = Pose()
    pose_msg.position.x = coord.position[0]
    pose_msg.position.y = coord.position[1]
    pose_msg.position.z = coord.position[2]
    pose_msg.orientation.x = coord.quaternion[0]
    pose_msg.orientation.y = coord.quaternion[1]
    pose_msg.orientation.z = coord.quaternion[2]
    pose_msg.orientation.w = coord.quaternion[3]
    return pose_msg


class YcbObject(enum.Enum):
    CRACKER_BOX = 2
    SUGAR_BOX = 3
    MUSTARD = 5
    PITCHER = 11
    CLEANSER = 12
    DRILL = 15
    WOOD_BLOCK = 16


class DemoInterface:

    # -------------------------------------------------------------------------
    # core
    # -------------------------------------------------------------------------

    joint_positions = dict(
        reset=[
            -0.061357464641332626,
            -1.0520118474960327,
            -0.03373617306351662,
            -1.7973287105560303,
            -0.009020943194627762,
            0.7718567252159119,
            0.8123827576637268,
        ],
        pre_place=[
            0.6948146820068359,
            -1.5501205921173096,
            -0.7638649344444275,
            -1.9011520147323608,
            -1.2214274406433105,
            0.8248165845870972,
            0.3954433798789978,
        ],
        pre_grasp=[
            1.2600178718566895,
            -1.135872721672058,
            -0.9610440135002136,
            -1.737254023551941,
            -1.0299488306045532,
            1.1363682746887207,
            1.001794457435608,
        ],
        pre_grasp_left=[
            -0.5062108039855957,
            -1.2614984512329102,
            0.818638801574707,
            -2.2020039558410645,
            0.8550509214401245,
            1.3013893365859985,
            0.9626140594482422,
        ],
    )

    def get_transform(self, target_frame, source_frame, time=rospy.Time(0)):
        """Get transform from TF.

        T_ee2base = self._get_transform(
            target_frame="panda_link0", source_frame="panda_suction_cup"
        )
        """
        position, quaternion = self._tf_listener.lookupTransform(
            target_frame, source_frame, time
        )
        return mercury.geometry.transformation_matrix(position, quaternion)

    def get_cartesian_path(
        self, av=None, coords=None, rotation_axis=True, steps=10
    ):
        if not (av is None) ^ (coords is None):
            raise ValueError("Either av or coords must be given")

        av_final = av

        av1 = self.robot.angle_vector()
        c1 = self.robot.rarm.end_coords.copy_worldcoords()

        if av is None:
            av = self.robot.rarm.inverse_kinematics(
                coords, rotation_axis=rotation_axis
            )
            if av is False:
                raise RuntimeError("IK failure")
        else:
            self.robot.angle_vector(av)

        c2 = self.robot.rarm.end_coords.copy_worldcoords()

        self.robot.angle_vector(av1)

        avs = []
        for p in np.linspace(0, 1, num=steps):
            c = skrobot.coordinates.midcoords(p, c1, c2)
            av = self.robot.rarm.inverse_kinematics(
                c, rotation_axis=rotation_axis
            )
            if av is not False:
                avs.append(av)
        if av_final is not None:
            avs.append(av_final)
        return avs

    def real2robot(self):
        self.ri.update_robot_state()
        self.robot.angle_vector(self.ri.potentio_vector())
        # self._viewer.redraw()

    def wait_interpolation(self):
        self.ri.wait_interpolation()
        self.real2robot()

    def send_avs(self, avs, time_scale=None, wait=True):
        self.recover_from_error()
        if time_scale is None:
            time_scale = 10
        self.ri.update_robot_state()
        av_delta = np.linalg.norm(avs - self.ri.potentio_vector(), axis=1)
        if (np.rad2deg(av_delta) < 5).all():
            self.ri.angle_vector(avs[-1], time_scale=10)
        else:
            self.ri.angle_vector_sequence(avs, time_scale=time_scale)
        if wait:
            self.wait_interpolation()

    def start_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(SetBoolRequest(data=True))

    def stop_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(SetBoolRequest(data=False))

    def recover_from_error(self):
        client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        client.wait_for_server()
        goal = ErrorRecoveryGoal()
        state = client.send_goal_and_wait(goal)
        succeeded = state == GoalStatus.SUCCEEDED

        if succeeded:
            rospy.loginfo("Recovered from error")
        else:
            rospy.logerr("Failed to recover from error")
        return succeeded

    # -------------------------------------------------------------------------
    # init
    # -------------------------------------------------------------------------

    def __init__(self):
        rospy.init_node("demo_interface")

        self._pub_heightmap = rospy.Publisher(
            "~debug/heightmap", Image, queue_size=1
        )
        self._pub_remove = rospy.Publisher(
            "/object_mapping/input/remove", ObjectClassArray, queue_size=1
        )

        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(60)
        )

        self.robot = Panda()
        self.robot.rarm.end_coords.translate([0, 0, 0.015], wrt="local")
        self.ri = skrobot.interfaces.PandaROSRobotInterface(robot=self.robot)
        # self._viewer = skrobot.viewers.TrimeshSceneViewer()
        # self._viewer.add(self.robot)
        # self._viewer.show()
        self.real2robot()

        self.env = _env.PickFromPileEnv()
        self.agent = _agent.DqnAgent(env=self.env, model="fusion_net")
        self.agent.build(training=False)
        self.agent.load_weights(
            "logs/20210709_005731-fusion_net-noise/weights/84500"
        )

        self.set_target(None)

        # publish placeholder
        bridge = cv_bridge.CvBridge()
        S = self.env.HEIGHTMAP_IMAGE_SIZE
        heightmap_msg = bridge.cv2_to_imgmsg(
            np.zeros((S, S * 2), dtype=np.uint8), encoding="mono8"
        )
        self._pub_heightmap.publish(heightmap_msg)

    # -------------------------------------------------------------------------
    # sensor inputs
    # -------------------------------------------------------------------------

    def set_target(self, target_object):
        self.target_object = target_object
        self.obs = {
            "pcd_cam": None,
            "pcd_base": None,
            "label_ins": None,
            "target_ins_id": None,
        }

    def _subscribe(self):
        sub_class = message_filters.Subscriber(
            "/camera/octomap_server/output/class",
            ObjectClassArray,
        )
        sub_depth = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
        )
        sub_label = message_filters.Subscriber(
            "/camera/octomap_server/output/label_rendered",
            Image,
        )
        self._subscribers = [sub_class, sub_depth, sub_label]
        sync = message_filters.ApproximateTimeSynchronizer(
            self._subscribers, queue_size=100, slop=0.1
        )
        sync.registerCallback(self._callback)

    def _unsubsribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, class_msg, depth_msg, label_ins_msg):
        cam_info_msg = rospy.wait_for_message(
            "/camera/aligned_depth_to_color/camera_info", CameraInfo
        )
        K = np.array(cam_info_msg.K).reshape(3, 3)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        bridge = cv_bridge.CvBridge()

        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan
        pcd_cam = mercury.geometry.pointcloud_from_depth(
            depth, fx=fx, fy=fy, cx=cx, cy=cy
        )
        pos, qua = self._tf_listener.lookupTransform(
            "panda_link0", "camera_color_optical_frame", time=rospy.Time(0)
        )
        T_cam2base = mercury.geometry.transformation_matrix(pos, qua)
        pcd_base = mercury.geometry.transform_points(pcd_cam, T_cam2base)
        self.obs["pcd_cam"] = pcd_cam
        self.obs["pcd_base"] = pcd_base

        self.obs["label_ins"] = bridge.imgmsg_to_cv2(label_ins_msg)

        for cls in class_msg.classes:
            if cls.class_id == self.target_object.value:
                break
        else:
            rospy.logerr(
                f"Failed to find the target object: {self.target_object}"
            )
            return
        self.obs["target_ins_id"] = cls.instance_id

    # -------------------------------------------------------------------------
    # actions
    # -------------------------------------------------------------------------

    def _check_observation(self):
        if any(value is None for value in self.obs.values()):
            rospy.logwarn("Please call capture_visual_observation() first")
            return False
        return True

    def start_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/request", Empty)
            client.call()

    def stop_passthrough(self):
        passthroughs = [
            "/camera/color/image_rect_color_passthrough",
        ]
        for passthrough in passthroughs:
            client = rospy.ServiceProxy(passthrough + "/stop", Empty)
            client.call()

    def capture_visual_observation(self):
        if self.target_object is None:
            rospy.logerr("self.target_object is not set")
            return

        self.set_target(self.target_object)  # clear obs

        self.start_passthrough()

        self._subscribe()
        while any(value is None for value in self.obs.values()):
            rospy.loginfo_throttle(10, "Waiting for visual observation")
            print(self.obs)
            rospy.timer.sleep(0.1)
        rospy.loginfo("Received visual observation")
        self._unsubsribe()

        self.stop_passthrough()

    def go_to_reset_pose(self):
        avs = self.get_cartesian_path(av=self.joint_positions["reset"])
        self.send_avs(avs, time_scale=3)

    def go_to_overlook_pose(self, xy=None, time_scale=5):
        if xy is None:
            xy = [0.5, 0]
        position = [xy[0], xy[1], 0.7]

        T_ee2cam = self.get_transform(
            target_frame="camera_color_optical_frame",
            source_frame="panda_suction_cup",
        )
        T_cam2base = mercury.geometry.transformation_matrix(
            position,
            mercury.geometry.quaternion_from_euler([3.14, 0, 1.57]),
        )

        T_ee2base = T_cam2base @ T_ee2cam

        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base)
        )
        self.send_avs(avs, time_scale=time_scale)

    def scan(self):
        T_ee2cam = self.get_transform(
            target_frame="camera_color_optical_frame",
            source_frame="panda_suction_cup",
        )
        T_cam2base = mercury.geometry.Coordinate(
            position=[0.5, -0.2, 0.6],
            quaternion=mercury.geometry.quaternion_from_euler(
                [np.deg2rad(190), 0, np.deg2rad(180)]
            ),
        ).matrix
        T_ee2base = T_cam2base @ T_ee2cam
        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base)
        )
        self.send_avs(avs, time_scale=5)

        self.start_passthrough()

        rospy.wait_for_message(
            "/singleview_3d_pose_estimation/output", ObjectPoseArray
        )

        avs_scan = []

        T_cam2base = mercury.geometry.Coordinate(
            position=[0.5, 0, 0.7],
            quaternion=mercury.geometry.quaternion_from_euler(
                [np.deg2rad(180), 0, np.deg2rad(180)]
            ),
        ).matrix
        T_ee2base = T_cam2base @ T_ee2cam
        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base)
        )
        avs_scan.extend(avs)

        T_cam2base = mercury.geometry.Coordinate(
            position=[0.5, 0.3, 0.6],
            quaternion=mercury.geometry.quaternion_from_euler(
                [np.deg2rad(170), 0, np.deg2rad(180)]
            ),
        ).matrix
        T_ee2base = T_cam2base @ T_ee2cam
        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base)
        )
        avs_scan.extend(avs)

        self.send_avs(avs_scan, time_scale=20)

        rospy.wait_for_message(
            "/singleview_3d_pose_estimation/output", ObjectPoseArray
        )

        self.stop_passthrough()

    def place(self):
        av = self.joint_positions["pre_place"]
        self.send_avs([av], time_scale=5)

        position = self.robot.panda_target_box.worldpos().copy()
        position[2] = 0.5
        T_ee2base = mercury.geometry.Coordinate(
            position=position,
            quaternion=mercury.geometry.quaternion_from_euler([3.14, 0, 0]),
        )
        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base.matrix),
            rotation_axis="z",
        )
        self.send_avs(avs, time_scale=10)

        T_ee2base.translate([0, 0, 0.3], wrt="local")
        avs = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(T_ee2base.matrix),
            rotation_axis="z",
        )
        self.send_avs(avs, time_scale=10)

        self.stop_grasp()
        rospy.timer.sleep(5)

        self.send_avs(avs[::-1], time_scale=5)

        self.go_to_reset_pose()

    def go_to_canonical_view(self):
        object_poses_msg = self._object_poses_msg
        object_centroids_msg = self._object_centroids_msg

        T_cam2base = self.get_transform(
            "panda_link0",
            object_poses_msg.header.frame_id,
            time=object_poses_msg.header.stamp,
        )

        if 0:
            for object_pose in object_centroids_msg.poses:
                if object_pose.class_id == self.target_object.value:
                    break
            anchor = [
                object_pose.pose.position.x,
                object_pose.pose.position.y,
                object_pose.pose.position.z,
            ]  # in cam
            anchor = mercury.geometry.transform_points([anchor], T_cam2base)[
                0
            ]  # in base
        else:
            for object_pose in object_poses_msg.poses:
                if object_pose.class_id == self.target_object.value:
                    break
            anchor = [
                object_pose.pose.position.x,
                object_pose.pose.position.y,
                object_pose.pose.position.z,
            ]  # in base

        # self._is_right = centroid[2] <= 0  # y
        self._is_right = True

        self.go_to_overlook_pose(xy=anchor[:2], time_scale=5)

    def pick_and_place(self):
        coord = self.plan_grasping()

        grasp_coord = coord

        aabb = np.array(
            [
                grasp_coord.position - self.env.HEIGHTMAP_SIZE / 2,
                grasp_coord.position + self.env.HEIGHTMAP_SIZE / 2,
            ]
        )
        aabb[0][2] = 0.05  # original: -0.05
        aabb[1][2] = 0.5
        heightmap, _, idmap = _get_heightmap.get_heightmap(
            self.obs["pcd_base"],
            np.zeros(self.obs["pcd_base"].shape, dtype=np.uint8),
            ids=self.obs["label_ins"] + 1,
            aabb=aabb,
            pixel_size=self.env.HEIGHTMAP_PIXEL_SIZE,
        )
        self.obs["heightmap"] = heightmap
        self.obs["maskmap"] = idmap == (self.obs["target_ins_id"] + 1)

        coord.translate([0, 0, -0.20], wrt="local")

        avs_pregrasp = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(coord.matrix),
            rotation_axis="z",
        )
        self.send_avs(avs_pregrasp, time_scale=7)

        coord.translate([0, 0, 0.15], wrt="local")

        avs_pregrasp = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(coord.matrix),
            rotation_axis="z",
        )
        self.send_avs(avs_pregrasp, time_scale=7, wait=False)

        coord.translate([0, 0, 0.05], wrt="local")

        avs_grasp = self.get_cartesian_path(
            coords=skrobot_coords_from_matrix(coord.matrix),
            rotation_axis="z",
        )

        avs_manipulation = self.plan_manipulation(grasp_coord)
        self.wait_interpolation()

        self.send_avs(avs_grasp, time_scale=10)
        self.start_grasp()

        self.send_avs(avs_manipulation, time_scale=30)
        self.send_avs([self.joint_positions["reset"]], time_scale=5)

        self.place()

    def plan_manipulation(self, grasp_coord):
        obs = self.obs.copy()
        object_poses_msg = self._object_poses_msg
        assert object_poses_msg.header.frame_id == "panda_link0"

        heightmap = obs["heightmap"]
        maskmap = obs["maskmap"]

        bridge = cv_bridge.CvBridge()
        heightmap_msg = bridge.cv2_to_imgmsg(
            imgviz.tile(
                [
                    imgviz.depth2rgb(heightmap, min_value=0, max_value=0.5),
                    np.uint8(maskmap * 255),
                ],
                shape=(1, 2),
                border=(255, 255, 255),
            ),
            encoding="rgb8",
        )
        self._pub_heightmap.publish(heightmap_msg)

        num_instance = len(object_poses_msg.poses)
        grasp_flags = np.zeros((num_instance,), dtype=np.uint8)
        object_labels = np.zeros(
            (num_instance, len(self.env.CLASS_IDS)), dtype=np.int8
        )
        object_poses = np.zeros((num_instance, 7), dtype=np.float32)

        for i, object_pose_msg in enumerate(object_poses_msg.poses):
            grasp_flags[i] = (
                object_pose_msg.instance_id == obs["target_ins_id"]
            )
            object_label = self.env.CLASS_IDS.index(object_pose_msg.class_id)
            object_labels[i] = np.eye(len(self.env.CLASS_IDS))[object_label]
            object_poses[i] = [
                object_pose_msg.pose.position.x - grasp_coord.position[0],
                object_pose_msg.pose.position.y - grasp_coord.position[1],
                object_pose_msg.pose.position.z - 0.1,
                object_pose_msg.pose.orientation.x,
                object_pose_msg.pose.orientation.y,
                object_pose_msg.pose.orientation.z,
                object_pose_msg.pose.orientation.w,
            ]
            if grasp_flags[i]:
                cls_msg = MFObjectClassArray()
                cls_msg.header = object_poses_msg.header
                cls_msg.classes = [
                    MFObjectClass(
                        instance_id=object_pose_msg.instance_id,
                        class_id=object_pose_msg.class_id,
                    )
                ]
                self._pub_remove.publish(cls_msg)

        ee_poses = np.zeros((self.env.episode_length, 7), dtype=np.float32)
        ee_poses = np.r_[
            ee_poses[1:],
            (
                np.hstack(grasp_coord.pose)
                - [
                    grasp_coord.position[0],
                    grasp_coord.position[1],
                    0.1,
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

        avs = []
        for i in range(self.env.episode_length):
            with torch.no_grad():
                q = self.agent.q(observation)

            q = q[0].cpu().numpy().reshape(-1)
            actions = np.argsort(q)[::-1]

            for action in actions:
                a = action // 2
                if i == self.env.episode_length - 1:
                    t = 1
                else:
                    t = action % 2
                dx, dy, dz, da, db, dg = self.env.actions[a]

                coord = mercury.geometry.Coordinate.from_matrix(
                    matrix=self.robot.rarm.end_coords.worldcoords().T()
                )

                coord.translate([dx, dy, dz], wrt="world")
                coord.rotate([da, db, dg], wrt="world")

                av_i = self.robot.rarm.inverse_kinematics(
                    skrobot_coords_from_matrix(coord.matrix)
                )
                if av_i is not False:
                    break
            avs.append(av_i)

            if t == 1:
                break

            ee_poses = np.r_[
                ee_poses[1:],
                (
                    np.hstack(coord.pose)
                    - [
                        grasp_coord.position[0],
                        grasp_coord.position[1],
                        0.1,
                        0,
                        0,
                        0,
                        0,
                    ]
                )[None],
            ].astype(np.float32)
            observation["ee_poses"] = torch.as_tensor(ee_poses)[None]

        return avs

    def plan_grasping(self):
        if not self._check_observation():
            return

        pcd_base = self.obs["pcd_base"]
        target_mask = self.obs["label_ins"] == self.obs["target_ins_id"]

        if self._is_right:
            av = self.joint_positions["pre_grasp"]
        else:
            av = self.joint_positions["pre_grasp_left"]
        self.send_avs([av], time_scale=3)

        T_base2ee = self.robot.rarm.end_coords.worldcoords().T()
        T_ee2base = np.linalg.inv(T_base2ee)

        pcd_ee = mercury.geometry.transform_points(pcd_base, T_base2ee)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normals_ee = mercury.geometry.normals_from_pointcloud(pcd_ee)

        r, c = scipy.ndimage.center_of_mass(target_mask)
        r, c = int(round(r)), int(round(c))

        rr, cc = skimage.draw.disk((r, c), radius=5, shape=target_mask.shape)
        grasp_position_mask = np.zeros_like(target_mask)
        grasp_position_mask[rr, cc] = True

        position = pcd_ee[r, c]
        normal = normals_ee[r, c]
        quaternion = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, 1], normal
        )
        coord = mercury.geometry.Coordinate(
            position=position,
            quaternion=quaternion,
        )
        coord.transform(T_ee2base, wrt="world")

        return coord

    def run(self, target_objects):
        if not self.recover_from_error():
            return

        self.real2robot()
        self.go_to_reset_pose()

        self.scan()

        for target_object in target_objects:
            self._object_poses_msg = rospy.wait_for_message(
                "/object_mapping/output/poses", ObjectPoseArray
            )
            self._object_centroids_msg = rospy.wait_for_message(
                "/object_mapping/grids_to_centroids/output/object_poses",
                ObjectPoseArray,
            )

            self.set_target(target_object)
            self.go_to_canonical_view()
            self.capture_visual_observation()

            self.pick_and_place()


if __name__ == "__main__":
    di = DemoInterface()
    IPython.embed()  # XXX
