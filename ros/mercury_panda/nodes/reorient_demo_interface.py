#!/usr/bin/env python

import enum
import itertools
import sys
import tempfile

import IPython
import numpy as np
import open3d
import path
import pybullet_planning as pp
import skrobot
import trimesh

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
import _utils  # NOQA
from pickable_eval import get_goal_oriented_reorient_poses  # NOQA
from reorient_dynamic import plan_dynamic_reorient  # NOQA


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


def tsdf_from_depth(depth, camera_to_base, K):
    T_camera_to_base = mercury.geometry.transformation_matrix(*camera_to_base)
    volume = open3d.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.04,
        color_type=open3d.integration.TSDFVolumeColorType.Gray32,
    )
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(depth),
        open3d.geometry.Image((depth * 1000).astype(np.uint16)),
        depth_trunc=1.0,
    )
    volume.integrate(
        rgbd,
        open3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        ),
        np.linalg.inv(T_camera_to_base),
    )
    mesh = volume.extract_triangle_mesh()
    mesh = mesh.simplify_vertex_clustering(voxel_size=0.02)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.r_[faces, faces[:, ::-1]],
    )
    return mesh


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
                for j_new in np.linspace(j_prev, j, num=10):
                    j_new = np.clip(j_new, lower, upper)
                    js_interpolated.append(j_new)
            j_prev = j
        return np.array(js_interpolated)

    def send_avs(self, avs, time_scale=10, wait=True):
        if not self.recover_from_error():
            return
        self.ri.update_robot_state()

        avs = self.interpolate_js(avs)

        av_prev = self.ri.potentio_vector()
        avs_filtered = []
        for i, av in enumerate(avs):
            av_delta = np.linalg.norm(av - av_prev)
            if av_delta > np.deg2rad(5):
                avs_filtered.append(av)
                av_prev = av
            elif i == len(avs) - 1:
                if avs_filtered:
                    # replace the last av
                    avs_filtered[-1] = av
        if avs_filtered:
            self.env.ri.setj(avs_filtered[-1])
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
        self._initialized = False

        rospy.init_node("demo_interface")

        self._tf_listener = tf.listener.TransformListener(
            cache_time=rospy.Duration(60)
        )

        self.ri = skrobot.interfaces.PandaROSRobotInterface(robot=Panda())

        self.env = Env(
            class_ids=None,
            real=True,
            robot_model="franka_panda/panda_drl",
            debug=False,
        )
        self.env.reset()

        self.real2robot()

    # -------------------------------------------------------------------------
    # sensor inputs
    # -------------------------------------------------------------------------

    def _subscribe_geometry(self):
        obs_keys = ["depth", "K", "camera_to_base"]
        self.obs = {key: None for key in obs_keys}

        sub_depth = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self._callback,
        )
        self._subscribers = [sub_depth]

    def _subscribe(self):
        obs_keys = ["depth", "K", "camera_to_base", "segm", "classes", "poses"]
        self.obs = {key: None for key in obs_keys}

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

    def _callback(
        self, depth_msg, class_msg=None, label_ins_msg=None, pose_msg=None
    ):
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
        self.obs["camera_to_base"] = np.hstack(
            self._tf_listener.lookupTransform(
                "panda_link0", "camera_color_optical_frame", time=rospy.Time(0)
            )
        )

        if class_msg is not None:
            self.obs["classes"] = class_msg.classes
        if label_ins_msg is not None:
            self.obs["segm"] = bridge.imgmsg_to_cv2(label_ins_msg)
        if pose_msg is not None:
            self.obs["poses"] = pose_msg.poses

    # -------------------------------------------------------------------------
    # actions
    # -------------------------------------------------------------------------

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

    def capture_geometry(self):
        self._subscribe_geometry()
        for i in range(30):
            rospy.loginfo_throttle(10, "Waiting for observation")
            rospy.sleep(0.1)
            if all(
                self.obs[key] is not None
                for key in ["depth", "K", "camera_to_base"]
            ):
                rospy.loginfo("Received observation")
                self._unsubsribe()
                break
        else:
            rospy.logerr("Timeout")

    def geometry_to_env(self):
        K = self.obs["K"]
        depth = self.obs["depth"]
        camera_to_base = np.hsplit(self.obs["camera_to_base"], [3])
        tsdf = tsdf_from_depth(depth, camera_to_base, K)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visual_file = path.Path(tmp_dir) / "bg_structure.obj"
            tsdf.export(visual_file)
            collision_file = mercury.pybullet.get_collision_file(
                visual_file, resolution=10000
            )
            mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                rgba_color=(0.5, 0.5, 0.5, 1),
            )

    def scan_geometry(self):
        self.capture_geometry()
        self.geometry_to_env()

    def capture_obs(self):
        if not self._initialized:
            rospy.logerr("Task is not initialized yet")
            return

        self.start_passthrough()

        self._subscribe()
        for i in range(30):
            rospy.loginfo_throttle(10, "Waiting for observation")
            rospy.sleep(0.1)
            if all(value is not None for value in self.obs.values()):
                rospy.loginfo("Received observation")
                self.stop_passthrough()
                self._unsubsribe()
                break
        else:
            rospy.logerr("Timeout")

    def obs_to_env(self):
        camera_to_base = np.hsplit(self.obs["camera_to_base"], [3])
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
                if self.env.fg_object_id is not None:
                    pp.remove_body(self.env.fg_object_id)
                object_id = mercury.pybullet.create_mesh_body(
                    visual_file=visual_file,
                    collision_file=collision_file,
                    position=obj_to_base[0],
                    quaternion=obj_to_base[1],
                    mass=0.1,
                )
                break
        target_instance_id = object_pose.instance_id

        K = self.obs["K"]
        depth = self.obs["depth"].copy()
        segm = self.obs["segm"]
        depth[segm == target_instance_id] = np.nan
        tsdf = tsdf_from_depth(depth, camera_to_base, K)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visual_file = path.Path(tmp_dir) / "bg_structure.obj"
            tsdf.export(visual_file)
            collision_file = mercury.pybullet.get_collision_file(
                visual_file, resolution=10000
            )
            bg_structure = mercury.pybullet.create_mesh_body(
                visual_file=visual_file,
                collision_file=collision_file,
                rgba_color=(0.5, 0.5, 0.5, 1),
            )

        self.env.bg_objects.append(bg_structure)
        self.env.object_ids = [object_id]
        self.env.fg_object_id = object_id
        self.env.update_obs()

    def reset_pose(self, cartesian=True):
        if cartesian:
            avs = self.env.ri.get_cartesian_path(j=self.env.ri.homej)
        else:
            avs = [self.env.ri.homej]
        self.send_avs(avs, time_scale=5)

    def look_at(self, eye, target, rotation_axis=True):
        c = mercury.geometry.Coordinate.from_matrix(
            mercury.geometry.look_at(eye, target)
        )
        for _ in range(4):
            c.rotate([0, 0, np.pi / 2])
            if abs(c.euler[2] - np.pi / 2) < np.pi / 4:
                break
        j = self.env.ri.solve_ik(
            c.pose,
            move_target=self.env.ri.robot_model.camera_link,
            n_init=10,
            thre=0.01,
            rthre=np.deg2rad(10),
            rotation_axis=rotation_axis,
        )
        if j is None or not self.env.ri.validatej(j):
            rospy.logerr("j is not found or invalid")
            return
        self.send_avs([j], time_scale=5)

    # -------------------------------------------------------------------------

    def init(self, nth=1):
        shelf = _utils.create_shelf(X=0.29, Y=0.41, Z=0.285, N=2)
        c = mercury.geometry.Coordinate()
        c.rotate([0, 0, -np.pi / 2])
        c.translate([0.575, 0.45, self.env.TABLE_OFFSET], wrt="world")
        pp.set_pose(shelf, c.pose)
        self.env.bg_objects.append(shelf)

        fg_class_id = 2
        c = mercury.geometry.Coordinate(
            [0.415, 0.39, 0.43], _utils.get_canonical_quaternion(fg_class_id)
        )
        for _ in range(nth - 1):
            c.translate([0.065, 0, 0], wrt="world")
        # mercury.pybullet.create_mesh_body(
        #     visual_file=mercury.datasets.ycb.get_visual_file(
        #         fg_class_id
        #     ),
        #     position=c.position,
        #     quaternion=c.quaternion,
        # )
        # c.translate([0.07, 0, 0], wrt="world")
        place_pose = c.pose

        c.translate([0, -0.3, 0.1], wrt="world")
        pre_place_pose = c.pose

        self.env._fg_class_id = fg_class_id
        self.env._place_pose = place_pose
        self.env._pre_place_pose = pre_place_pose

        visual_file = mercury.datasets.ycb.get_visual_file(
            self.env._fg_class_id
        )
        self._obj_goal = mercury.pybullet.create_mesh_body(
            visual_file=visual_file,
            rgba_color=(0.5, 0.5, 0.5, 0.5),
            position=self.env.PLACE_POSE[0],
            quaternion=self.env.PLACE_POSE[1],
        )

        self._initialized = True

    def reset(self):
        self.env.reset()
        self.env._fg_class_id = None
        self.env._place_pose = None
        self.env._pre_place_pose = None

        pp.remove_body(self._obj_goal)
        self._obj_goal = None

        self._initialized = False

    def look_at_pile(self):
        self.look_at(eye=[0.5, 0, 0.7], target=[0.5, 0, 0])

    def scan_pile(self):
        if not self._initialized:
            rospy.logerr("Task is not initialized yet")
            return

        self.look_at_pile()
        self.capture_obs()
        self.obs_to_env()

    def look_at_reoriented(self):
        if self.env.fg_object_id is None:
            target = [0.5, -0.5, 0.1]
        else:
            target = pp.get_pose(self.env.fg_object_id)[0]
        self.look_at(
            eye=[target[0] - 0.3, -0.3, 0.7], target=target, rotation_axis="z"
        )

    def scan_reoriented(self):
        if not self._initialized:
            rospy.logerr("Task is not initialized yet")
            return

        self.look_at_reoriented()
        self.capture_obs()
        self.obs_to_env()

    def plan_reorient(self, heuristic=False):
        if heuristic:
            grasp_poses = _reorient.get_grasp_poses(self.env)
            grasp_poses = list(itertools.islice(grasp_poses, 12))
            reorient_poses = _reorient.get_static_reorient_poses(self.env)

            result = {}
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
        else:
            (
                reorient_poses,
                pickable,
                target_grasp_poses,
            ) = get_goal_oriented_reorient_poses(self.env)

            grasp_poses = _reorient.get_grasp_poses(self.env)  # in world
            grasp_poses = list(itertools.islice(grasp_poses, 25))

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
        return result

    def pick_and_reorient(self, heuristic=False):
        result = self.plan_reorient(heuristic=heuristic)
        if "js_place" not in result:
            rospy.logerr("Failed to plan reorientation")
            return

        self.send_avs(result["js_pre_grasp"], time_scale=5)
        self.wait_interpolation()

        js = self.env.ri.get_cartesian_path(j=result["j_grasp"])
        if _utils.get_class_id(self.env.fg_object_id) == 5:
            with pp.WorldSaver():
                self.env.ri.setj(js[-1])
                c = mercury.geometry.Coordinate(
                    *self.env.ri.get_pose("tipLink")
                )
                c.translate([0, 0, 0.02])
                j = self.env.ri.solve_ik(c.pose)
                if j is not None:
                    js = np.r_[js, [j]]
        self.send_avs(js)
        self.wait_interpolation()

        self.start_grasp()
        rospy.sleep(2)

        js = result["js_place"]
        with pp.WorldSaver():
            self.env.ri.setj(js[-1])
            c = mercury.geometry.Coordinate(*self.env.ri.get_pose("tipLink"))
            if _utils.get_class_id(self.env.fg_object_id) == 5:
                c.translate([0, 0, -0.05], wrt="world")
            else:
                c.translate([0, 0, -0.01], wrt="world")
            j = self.env.ri.solve_ik(c.pose)
            if j is not None:
                js = np.r_[js, [j]]
        self.send_avs(js, time_scale=5)
        self.wait_interpolation()

        self.stop_grasp()
        rospy.sleep(5)

        js = result["js_post_place"]
        self.send_avs(js, time_scale=5)

        self.send_avs([self.env.ri.homej], time_scale=3)
        self.wait_interpolation()

        pp.set_pose(
            self.env.fg_object_id, np.hsplit(result["reorient_pose"], [3])
        )

    def plan_place(self):
        pcd_in_obj, normals_in_obj = _reorient.get_query_ocs(self.env)
        dist_from_centroid = np.linalg.norm(pcd_in_obj, axis=1)

        indices = np.arange(pcd_in_obj.shape[0])
        p = dist_from_centroid.max() - dist_from_centroid

        keep = dist_from_centroid < np.median(dist_from_centroid)
        indices = indices[keep]
        p = p[keep]
        indices = np.random.choice(indices, 5, p=p / p.sum())

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

        result = _reorient.plan_place(self.env, grasp_poses)

        return result

    def pick_and_place(self):
        result = self.plan_place()
        if "js_place" not in result:
            rospy.logerr("Failed to plan placement")
            return

        self.send_avs(result["js_pre_grasp"], time_scale=5)
        self.wait_interpolation()

        self.send_avs(
            self.env.ri.get_cartesian_path(j=result["j_grasp"]),
            time_scale=20,
        )
        self.wait_interpolation()

        self.start_grasp()
        rospy.sleep(2)

        self.send_avs(result["js_pre_place"], time_scale=5)
        self.wait_interpolation()

        self.send_avs(result["js_place"], time_scale=20)
        self.wait_interpolation()

        self.stop_grasp()
        rospy.sleep(6)

        self.send_avs(result["js_place"][::-1], time_scale=5)
        self.wait_interpolation()

        self.reset_pose(cartesian=False)
        self.wait_interpolation()


if __name__ == "__main__":
    di = ReorientDemoInterface()
    di.sp = di.scan_pile
    di.sr = di.scan_reoriented
    di.pp = di.pick_and_place
    di.pr = di.pick_and_reorient
    di.rs = di.reset
    di.rp = di.reset_pose
    IPython.embed()
