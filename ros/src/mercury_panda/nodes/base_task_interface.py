#!/usr/bin/env python

import IPython
import numpy as np
import pybullet as p
import pybullet_planning as pp

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

        self._subscriber_base = MessageSubscriber(
            [
                ("/camera/color/camera_info", CameraInfo),
                ("/camera/color/image_rect_color", Image),
                ("/camera/aligned_depth_to_color/image_raw", Image),
            ],
            callback=self._subscriber_base_callback,
        )
        self._subscriber_base_points = None
        self._subscriber_base.subscribe()

    def _subscriber_base_callback(self, info_msg, rgb_msg, depth_msg):
        K = np.array(info_msg.K).reshape(3, 3)
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        try:
            camera_to_base = self.lookup_transform(
                "panda_link0",
                info_msg.header.frame_id,
                time=info_msg.header.stamp,
            )
        except tf.ExtrapolationException:
            return

        pcd = mercury.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd = mercury.geometry.transform_points(
            pcd, mercury.geometry.transformation_matrix(*camera_to_base)
        )

        subscriber_base_points = _pybullet.draw_points(pcd, rgb, size=1)

        if self._subscriber_base_points is not None:
            pp.remove_debug(self._subscriber_base_points)
        self._subscriber_base_points = subscriber_base_points

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

    def lookup_transform(self, target_frame, source_frame, time, timeout=None):
        if timeout is not None:
            self._tf_listener.waitForTransform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=time,
                timeout=timeout,
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

    def movejs(self, js, time_scale=None, wait=True, max_accel=None):
        if not self.recover_from_error():
            return
        if time_scale is None:
            time_scale = 3
        if max_accel is None:
            max_accel = 1
        js = np.asarray(js)
        self.real2robot()
        self.ri.angle_vector_sequence(
            js, time_scale=time_scale, max_accel=max_accel
        )
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

    def look_at_pile(self, *args, **kwargs):
        self.look_at(eye=[0.5, 0, 0.7], target=[0.5, 0, 0], *args, **kwargs)

    def init_workspace(self):
        # light
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, True, lightPosition=(100, -100, 0.5)
        )

        # table
        pp.set_texture(self._env.plane)

        # left wall
        obj = pp.create_box(w=3, l=0.01, h=1.05, color=(0.6, 0.6, 0.6, 1))
        pp.set_pose(
            obj,
            (
                (-0.0010000000000000002, 0.6925000000000028, 0.55),
                (0.0, 0.0, 0.0194987642109932, 0.9998098810245096),
            ),
        )
        self._env.bg_objects.append(obj)

        # back wall
        obj = pp.create_box(w=0.01, l=3, h=1.05, color=(0.7, 0.7, 0.7, 1))
        pp.set_pose(obj, ([-0.4, 0, 1.05 / 2], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        # ceiling
        obj = pp.create_box(w=3, l=3, h=0.5, color=(1, 1, 1, 1))
        pp.set_pose(obj, ([0, 0, 0.25 + 1.05], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        # bin
        obj = mercury.pybullet.create_bin(
            X=0.3, Y=0.3, Z=0.11, color=(0.7, 0.7, 0.7, 1)
        )
        pp.set_pose(
            obj,
            (
                (0.4495000000000015, 0.5397000000000006, 0.059400000000000126),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        self._env.bg_objects.append(obj)
        self._bin = obj

        # _pybullet.annotate_pose(obj)


def main():
    rospy.init_node("base_task_interface")
    self = BaseTaskInterface()  # NOQA
    IPython.embed()


if __name__ == "__main__":
    main()
