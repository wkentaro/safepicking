#!/usr/bin/env python

import enum
import itertools
import sys
import tempfile

import numpy as np
import pybullet_planning as pp
import skrobot

import mercury

import actionlib
from actionlib_msgs.msg import GoalStatus
import cv_bridge
from franka_control.msg import ErrorRecoveryAction
from franka_control.msg import ErrorRecoveryGoal
import message_filters
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
from std_srvs.srv import SetBoolRequest
import tf

from mercury_panda.msg import ObjectClassArray
from morefusion_panda_ycb_video.msg import ObjectPoseArray

sys.path.insert(0, "../../../examples/reorient")

from _env import Env  # NOQA
import _reorient  # NOQA


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


def pose_from_msg(msg):
    position = (
        msg.position.x,
        msg.position.y,
        msg.position.z,
    )
    quaternion = (
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    )
    return position, quaternion


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

    def interpolate_js(self, js):
        lower, upper = self.env.ri.get_bounds()
        js_interpolated = []
        j_prev = None
        for j in js:
            if j_prev is None:
                js_interpolated.append(np.clip(j, lower, upper))
            else:
                for j_new in np.linspace(j_prev, j, num=5):
                    j_new = np.clip(j_new, lower, upper)
                    js_interpolated.append(j_new)
            j_prev = j
        return np.array(js_interpolated)

    def send_avs(self, avs, time_scale=None, wait=True):
        if not self.recover_from_error():
            return
        if time_scale is None:
            time_scale = 10
        self.ri.update_robot_state()

        avs = self.interpolate_js(avs)

        av_prev = self.ri.potentio_vector()
        avs_filtered = []
        for av in avs:
            av_delta = np.linalg.norm(av - av_prev)
            if av_delta > np.deg2rad(5):
                avs_filtered.append(av)
                av_prev = av
        self.ri.angle_vector_sequence(avs_filtered, time_scale=time_scale)
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

        self.env = Env(
            class_ids=None, real=True, robot_model="franka_panda/panda_drl"
        )
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

        sub_depth = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
        )
        sub_class = message_filters.Subscriber(
            "/camera/mask_rcnn_instance_segmentation/output/class",
            ObjectClassArray,
        )
        sub_label = message_filters.Subscriber(
            "/camera/mask_rcnn_instance_segmentation/output/label_ins",
            Image,
        )
        sub_pose = message_filters.Subscriber(
            "/singleview_3d_pose_estimation/output",
            ObjectPoseArray,
        )
        self._subscribers = [sub_depth, sub_class, sub_label, sub_pose]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def _unsubsribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, depth_msg, class_msg, label_ins_msg, pose_msg):
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
        self.obs["poses"] = pose_msg.poses

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

    def go_to_reset_pose(self, cartesian=True):
        if cartesian:
            avs = self.get_cartesian_path(av=self.env.ri.homej)
        else:
            avs = [self.env.ri.homej]
        self.send_avs(avs, time_scale=5)

    def go_to_overlook_pose(self, position=None):
        if position is None:
            position = [0.5, 0, 0.7]

        pose = self.env.ri.get_pose("camera_link")
        j = self.env.ri.solve_ik(
            (position, pose[1]),
            move_target=self.env.ri.robot_model.camera_link,
            rotation_axis="z",
        )
        js = self.get_cartesian_path(av=j)
        self.send_avs(js, time_scale=5)

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
        mask = (segm == target_id) & (~np.isnan(self.obs["depth"]))
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

        return np.hstack([pcd_in_base, quaternion_in_base])

    # -------------------------------------------------------------------------

    def init_task(self):
        self.env._fg_class_id = 2
        c = mercury.geometry.Coordinate([0.5, 0.52, 0.54], [0, 0, 0, 1])
        c.rotate([0, 0, np.pi])
        self.env._place_pose = c.pose
        c.translate([0, -0.3, 0], wrt="world")
        self.env._pre_place_pose = c.pose

    def capture_to_reorient(self):
        self.go_to_overlook_pose()
        self.capture_visual_observation()
        self.observation_to_env()

    def observation_to_env(self):
        camera_to_base = self.obs["camera_to_base"]
        for object_pose in self.obs["poses"]:
            if object_pose.class_id == self.env._fg_class_id:
                obj_to_camera = pose_from_msg(object_pose.pose)
                obj_to_base = pp.multiply(camera_to_base, obj_to_camera)
                visual_file = mercury.datasets.ycb.get_visual_file(
                    class_id=object_pose.class_id
                )
                collision_file = mercury.pybullet.get_collision_file(
                    visual_file
                )
                object_id = mercury.pybullet.create_mesh_body(
                    visual_file=visual_file,
                    collision_file=collision_file,
                    position=obj_to_base[0],
                    quaternion=obj_to_base[1],
                    mass=0.1,
                )
                break
        self.env.object_ids = [object_id]
        self.env.fg_object_id = object_id
        self.env.update_obs()

        mercury.pybullet.duplicate(
            self.env.fg_object_id,
            collision=False,
            rgba_color=(0.5, 0.5, 0.5, 0.5),
            mass=0,
            position=self.env.PLACE_POSE[0],
            quaternion=self.env.PLACE_POSE[1],
        )

    def pick_and_reorient_learned(self):
        from pickable_eval import get_goal_oriented_reorient_poses
        from reorient_dynamic import plan_dynamic_reorient

        (
            reorient_poses,
            pickable,
            target_grasp_poses,
        ) = get_goal_oriented_reorient_poses(self.env)

        grasp_poses = _reorient.get_grasp_poses(self.env)  # in world
        grasp_poses = list(itertools.islice(grasp_poses, 100))

        for threshold in np.linspace(0.9, 0.1, num=10):
            indices = np.where(pickable > threshold)[0]
            if indices.size > 100:
                break
        indices = np.random.choice(
            indices, min(indices.size, 1000), replace=False
        )
        reorient_poses = reorient_poses[indices]
        pickable = pickable[indices]

        result = plan_dynamic_reorient(
            self.env, grasp_poses, reorient_poses, pickable
        )

        if 0:
            _reorient.execute_reorient(self.env, result)
        else:
            self.send_avs(result["js_pre_grasp"], time_scale=5)
            self.wait_interpolation()

            self.send_avs(self.get_cartesian_path(av=result["j_grasp"]))
            self.wait_interpolation()

            self.start_grasp()
            rospy.sleep(2)

            js = result["js_place"]
            self.send_avs(js)
            self.wait_interpolation()

            self.stop_grasp()
            rospy.sleep(5)

            js = result["js_post_place"]
            self.send_avs(js, time_scale=5)
            self.wait_interpolation()

        pp.set_pose(
            self.env.fg_object_id, np.hsplit(result["reorient_pose"], [3])
        )

    def pick_and_reorient(self):
        if 0:
            pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(self.env)
            indices = np.random.permutation(pcd_in_obj.shape[0])[:20]
            pcd_in_obj = pcd_in_obj[indices]
            normals_in_obj = normals_in_obj[indices]
            quaternion_in_obj = mercury.geometry.quaternion_from_vec2vec(
                [0, 0, -1], normals_in_obj
            )
            grasp_poses = np.hstack([pcd_in_obj, quaternion_in_obj])  # in obj
        else:
            grasp_poses = _reorient.get_grasp_poses(self.env)  # in world
            grasp_poses = list(itertools.islice(grasp_poses, 12))

        reorient_poses = _reorient.get_static_reorient_poses(self.env)

        for grasp_pose, reorient_pose in itertools.product(
            grasp_poses, reorient_poses
        ):
            result = _reorient.plan_reorient(
                self.env, grasp_pose, reorient_pose
            )
            if "js_place" in result:
                break
        else:
            rospy.logerr("No solution found")
            return

        if 0:
            _reorient.execute_reorient(self.env, result)
        else:
            self.send_avs(result["js_pre_grasp"], time_scale=5)
            self.wait_interpolation()

            self.send_avs(self.get_cartesian_path(av=result["j_grasp"]))
            self.wait_interpolation()

            self.start_grasp()
            rospy.sleep(2)

            js = result["js_place"]
            self.send_avs(js)
            self.wait_interpolation()

            self.stop_grasp()
            rospy.sleep(5)

            js = result["js_post_place"]
            self.send_avs(js, time_scale=5)
            self.wait_interpolation()

        pp.set_pose(
            self.env.fg_object_id, np.hsplit(result["reorient_pose"], [3])
        )

    def capture_to_place(self):
        eye = [0.2, -0.3, 0.7]
        target = [0.5, -0.5, 0.1]
        camera_to_base = mercury.geometry.pose_from_matrix(
            mercury.geometry.look_at(eye, target)
        )
        j = self.env.ri.solve_ik(
            camera_to_base, move_target=self.env.ri.robot_model.camera_link
        )
        self.send_avs([j], time_scale=5)

        self.capture_visual_observation()
        self.observation_to_env()

    def pick_and_place(self):
        pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(self.env)
        indices = np.random.permutation(pcd_in_obj.shape[0])[:20]
        pcd_in_obj = pcd_in_obj[indices]
        normals_in_obj = normals_in_obj[indices]
        quaternion_in_obj = mercury.geometry.quaternion_from_vec2vec(
            [0, 0, -1], normals_in_obj
        )
        grasp_poses = np.hstack([pcd_in_obj, quaternion_in_obj])  # in obj

        result = _reorient.plan_place(self.env, grasp_poses)

        if "js_place" not in result:
            rospy.logerr("No solution found")
            return

        if 0:
            _reorient.execute_place(self.env, result)
        else:
            self.send_avs(result["js_pre_grasp"], time_scale=5)
            self.wait_interpolation()

            self.send_avs(
                self.get_cartesian_path(av=result["j_grasp"]), time_scale=20
            )
            self.wait_interpolation()

            self.start_grasp()
            rospy.sleep(2)

            self.send_avs(result["js_pre_place"], time_scale=5)
            self.wait_interpolation()

            self.send_avs(result["js_place"], time_scale=20)
            self.wait_interpolation()

            self.stop_grasp()
            rospy.sleep(5)

            self.send_avs(result["js_place"][::-1], time_scale=20)
            self.wait_interpolation()

            self.go_to_reset_pose(cartesian=False)
            self.wait_interpolation()


if __name__ == "__main__":
    di = ReorientDemoInterface()
    import IPython

    IPython.embed()  # NOQA
