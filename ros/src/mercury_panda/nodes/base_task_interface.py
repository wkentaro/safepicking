#!/usr/bin/env python

import IPython
import numpy as np

import mercury

import actionlib
from actionlib_msgs.msg import GoalStatus
import cv_bridge
from franka_msgs.msg import ErrorRecoveryAction
from franka_msgs.msg import ErrorRecoveryGoal
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
import tf

from mercury.examples.reorientation import _env

from _message_subscriber import MessageSubscriber
from _panda import Panda
from _panda_ros_robot_interface import PandaROSRobotInterface
import _pybullet


class BaseTaskInterface:
    def __init__(self):
        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(60)
        )

        self._ri = PandaROSRobotInterface(robot=Panda())

        self._env = _env.Env(
            class_ids=None,
            real=True,
            robot_model="franka_panda/panda_drl",
            debug=False,
        )
        self._env.reset()

        self.real2robot()

        self.subscriber_rgbd = MessageSubscriber(
            [
                ("/camera/color/camera_info", CameraInfo),
                ("/camera/color/image_rect_color", Image),
                ("/camera/aligned_depth_to_color/image_raw", Image),
            ]
        )

    @property
    def pi(self):
        return self._env.ri

    @property
    def ri(self):
        return self._ri

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

    def add_pointcloud_to_pybullet(self):
        self.start_passthrough()
        self.subscriber_rgbd.wait_for_messages()
        self.stop_passthrough()

        info_msg, rgb_msg, depth_msg = self.subscriber_rgbd.msgs

        K = np.array(info_msg.K).reshape(3, 3)
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        camera_to_base = self.lookup_transform(
            "panda_link0",
            info_msg.header.frame_id,
            time=info_msg.header.stamp,
        )

        pcd = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd = mercury.geometry.transform_points(
            pcd, mercury.geometry.transformation_matrix(*camera_to_base)
        )

        _pybullet.draw_points(pcd, rgb, size=3)

    def lookup_transform(self, target_frame, source_frame, time):
        self._tf_listener.waitForTransform(
            target_frame=target_frame,
            source_frame=source_frame,
            time=time,
            timeout=rospy.Duration(0.1),
        )
        return self._tf_listener.lookupTransform(
            target_frame=target_frame, source_frame=source_frame, time=time
        )

    def recover_from_error(self):
        client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        client.wait_for_server()

        if client.get_state() == GoalStatus.SUCCEEDED:
            return True

        goal = ErrorRecoveryGoal()
        state = client.send_goal_and_wait(goal)
        succeeded = state == GoalStatus.SUCCEEDED

        if succeeded:
            rospy.loginfo("Recovered from error")
        else:
            rospy.logerr("Failed to recover from error")
        return succeeded

    def real2robot(self):
        self.ri.update_robot_state()
        self.pi.setj(self.ri.potentio_vector())
        for attachment in self.pi.attachments:
            attachment.assign()

    def movejs(self, js, time_scale=None, wait=True):
        if not self.recover_from_error():
            return
        if time_scale is None:
            time_scale = 10
        js = np.asarray(js)
        self.real2robot()
        self.ri.angle_vector_sequence(js, time_scale=time_scale, max_accel=1)
        if wait:
            self.wait_interpolation()

    def wait_interpolation(self):
        controller_actions = self.ri.controller_table[self.ri.controller_type]
        while True:
            states = [action.get_state() for action in controller_actions]
            if all(s >= GoalStatus.SUCCEEDED for s in states):
                break
            self.real2robot()
            rospy.sleep(0.01)
        if not all(s == GoalStatus.SUCCEEDED for s in states):
            rospy.logwarn("Some joint control requests have failed")
            return False
        return True

    def start_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(data=True)

    def stop_grasp(self):
        client = rospy.ServiceProxy("/set_suction", SetBool)
        client.call(data=False)

    def reset_pose(self, *args, **kwargs):
        self.movejs([self.pi.homej], *args, **kwargs)

    def _solve_ik_for_look_at(self, eye, target, rotation_axis=True):
        c = mercury.geometry.Coordinate.from_matrix(
            mercury.geometry.look_at(eye, target)
        )
        if rotation_axis is True:
            for _ in range(4):
                c.rotate([0, 0, np.deg2rad(90)])
                if abs(c.euler[2] - np.deg2rad(-90)) < np.pi / 4:
                    break
        j = self.pi.solve_ik(
            c.pose,
            move_target=self.pi.robot_model.camera_link,
            n_init=20,
            thre=0.05,
            rthre=np.deg2rad(15),
            rotation_axis=rotation_axis,
            validate=True,
        )
        if j is None:
            rospy.logerr("j is not found")
            return
        return j

    def look_at(self, eye, target, rotation_axis=True, *args, **kwargs):
        j = self._solve_ik_for_look_at(eye, target, rotation_axis)
        self.movejs([j], *args, **kwargs)


def main():
    rospy.init_node("base_task_interface")
    self = BaseTaskInterface()  # NOQA
    IPython.embed()


if __name__ == "__main__":
    main()
