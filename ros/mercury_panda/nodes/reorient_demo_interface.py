#!/usr/bin/env python

import enum
import sys
import tempfile

import IPython
import numpy as np
import pybullet_planning as pp
import skrobot

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

sys.path.insert(0, "../../../examples/reorient")

from _env import Env  # NOQA


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
        model.end_coords = self.tipLink
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


class ReorientDemoInterface:

    # -------------------------------------------------------------------------
    # core
    # -------------------------------------------------------------------------

    def get_transform(self, target_frame, source_frame, time=rospy.Time(0)):
        """Get transform from TF.

        T_ee2base = self._get_transform(
            target_frame="panda_link0", source_frame="tipLink"
        )
        """
        position, quaternion = self._tf_listener.lookupTransform(
            target_frame, source_frame, time
        )
        return mercury.geometry.transformation_matrix(position, quaternion)

    def get_cartesian_path(
        self, av=None, pose=None, rotation_axis=True, steps=10
    ):
        if not (av is None) ^ (pose is None):
            raise ValueError("Either av or coords must be given")

        av1 = self.env.ri.getj()
        p1 = self.env.ri.get_pose("tipLink")

        if av is None:
            av = self.env.ri.solve_ik(pose, rotation_axis=rotation_axis)
            if av is None:
                raise RuntimeError("IK failure")
        else:
            self.env.ri.setj(av)
        av_final = av

        p2 = self.env.ri.get_pose("tipLink")

        self.env.ri.setj(av1)

        avs = []
        for p in pp.interpolate_poses_by_num_steps(p1, p2, num_steps=steps):
            av = self.env.ri.solve_ik(p, rotation_axis=rotation_axis)
            if av is not None:
                avs.append(av)
        if av_final is not None:
            avs.append(av_final)
        avs = np.array(avs)
        return avs

    def real2robot(self):
        self.ri.update_robot_state()
        self.env.ri.setj(self.ri.potentio_vector())

    def wait_interpolation(self):
        self.ri.wait_interpolation()
        self.real2robot()

    def send_avs(self, avs, time_scale=None, wait=True):
        if not self.recover_from_error():
            return
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

        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(60)
        )

        self.ri = skrobot.interfaces.PandaROSRobotInterface(robot=Panda())

        self.env = Env(class_ids=None, real=True)
        self.env.reset()

        self.real2robot()

    # -------------------------------------------------------------------------
    # sensor inputs
    # -------------------------------------------------------------------------

    def reset_obs(self):
        obs_keys = ["depth", "K", "camera_to_base", "segm", "classes"]
        self.obs = {key: None for key in obs_keys}

    def _subscribe(self):
        self.reset_obs()

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

        bridge = cv_bridge.CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        self.obs["depth"] = depth
        self.obs["K"] = K
        self.obs["camera_to_base"] = self._tf_listener.lookupTransform(
            "panda_link0", "camera_color_optical_frame", time=rospy.Time(0)
        )
        self.obs["segm"] = bridge.imgmsg_to_cv2(label_ins_msg)
        self.obs["classes"] = class_msg.classes

    # -------------------------------------------------------------------------
    # actions
    # -------------------------------------------------------------------------

    def _check_observation(self):
        if any(value is None for value in self.obs.values()):
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
        self.start_passthrough()

        self._subscribe()
        while not self._check_observation():
            rospy.loginfo_throttle(10, "Waiting for visual observation")
            rospy.timer.sleep(0.1)
        rospy.loginfo("Received visual observation")
        self._unsubsribe()

        self.stop_passthrough()

    def go_to_reset_pose(self):
        avs = self.get_cartesian_path(av=self.env.ri.homej)
        self.send_avs(avs, time_scale=3)

    def go_to_overlook_pose(self, xy=None, time_scale=5):
        if xy is None:
            xy = [0.5, 0]
        position = [xy[0], xy[1], 0.7]

        T_ee2cam = self.get_transform(
            target_frame="camera_color_optical_frame",
            source_frame="tipLink",
        )
        T_cam2base = mercury.geometry.transformation_matrix(
            position,
            mercury.geometry.quaternion_from_euler([3.14, 0, 1.57]),
        )

        T_ee2base = T_cam2base @ T_ee2cam

        avs = self.get_cartesian_path(
            pose=mercury.geometry.pose_from_matrix(T_ee2base)
        )
        self.send_avs(avs, time_scale=time_scale)

    def get_grasp_poses(self):
        if not self._check_observation():
            rospy.logerr("Please call capture_visual_observation() first")
            return

        for cls_msg in self.obs["classes"]:
            if cls_msg.class_id == 2:
                rospy.loginfo(f"Target object is found: {cls_msg}")
                break
        else:
            rospy.logerr(f"Target object is not found: {self.obs['classes']}")
            return

        target_id = cls_msg.instance_id
        segm = self.obs["segm"]
        depth = self.obs["depth"]

        K = self.obs["K"]
        mask = segm == target_id
        pcd_in_camera = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        normals_in_camera = mercury.geometry.normals_from_pointcloud(
            pcd_in_camera
        )
        pcd_in_camera = pcd_in_camera[mask]
        normals_in_camera = normals_in_camera[mask]

        camera_to_base = self.obs["camera_to_base"]
        pcd_in_base = mercury.geometry.transform_points(
            pcd_in_camera,
            mercury.geometry.transformation_matrix(*camera_to_base),
        )
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

        return pcd_in_base, quaternion_in_base


if __name__ == "__main__":
    di = ReorientDemoInterface()
    IPython.embed()  # XXX
