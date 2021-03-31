import itertools

import numpy as np
import path
import pybullet as p
import pybullet_planning

from .. import geometry
from .suction_gripper import SuctionGripper

import skrobot


here = path.Path(__file__).abspath().parent


class PandaRobotInterface:
    def __init__(self, pose=None):
        self.pose = pose

        urdf_file = here / "assets/franka_panda/panda_suction.urdf"
        self.robot_model = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=urdf_file
        )
        self.robot = pybullet_planning.load_pybullet(
            urdf_file, fixed_base=True
        )
        self.ee = pybullet_planning.link_from_name(self.robot, "tipLink")

        self.gripper = SuctionGripper(self.robot, self.ee)

        if self.pose is not None:
            pybullet_planning.set_pose(self.robot, self.pose)

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.robot)
        joints = [p.getJointInfo(self.robot, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.homej = [0, -np.pi / 4, 0, -np.pi / 2, 0, np.pi / 4, 0]
        for joint in self.joints:
            p.resetJointState(self.robot, joint, self.homej[joint])

    def world_to_base(self, a_to_world):
        if self.pose is None:
            a_to_base = a_to_world
        else:
            base_to_world = self.pose
            world_to_base = pybullet_planning.invert(base_to_world)
            a_to_base = pybullet_planning.invert(world_to_base, a_to_world)
        return a_to_base

    def setj(self, joint_positions):
        for joint, joint_position in zip(self.joints, joint_positions):
            p.resetJointState(self.robot, joint, joint_position)

    def getj(self):
        joint_positions = []
        for joint in self.joints:
            joint_positions.append(p.getJointState(self.robot, joint)[0])
        return joint_positions

    def movej(self, targj, speed=0.01):
        assert len(targj) == len(self.joints)
        for i in itertools.count():
            currj = [p.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains,
            )
            yield i

    def solve_ik(self, pose, **kwargs):
        c = geometry.Coordinate(*self.world_to_base(pose))
        joint_positions = self.get_skrobot().inverse_kinematics(
            c.skrobot_coords,
            move_target=self.robot_model.tipLink,
            **kwargs,
        )
        if joint_positions is False:
            return
        assert len(joint_positions) == len(self.joints)
        return joint_positions

    # def _solve_ik_pybullet(self, pose):
    #     n_joints = p.getNumJoints(self.robot)
    #     lower_limits = []
    #     upper_limits = []
    #     for i in range(n_joints):
    #         joint_info = p.getJointInfo(self.robot, i)
    #         lower_limits.append(joint_info[8])
    #         upper_limits.append(joint_info[9])
    #     joint_positions = p.calculateInverseKinematics(
    #         bodyUniqueId=self.robot,
    #         endEffectorLinkIndex=self.ee,
    #         targetPosition=pose[0],
    #         targetOrientation=pose[1],
    #         lowerLimits=lower_limits,
    #         upperLimits=upper_limits,
    #         restPoses=self.homej,
    #         maxNumIterations=1000,
    #         residualThreshold=1e-5,
    #     )
    #     return joint_positions
    #
    # def _solve_ik_pybullet_planning(self, pose):
    #     with pybullet_planning.LockRenderer():
    #         with pybullet_planning.WorldSaver():
    #             joint_positions = pybullet_planning.inverse_kinematics(
    #                 self.robot,
    #                 self.ee,
    #                 pose,
    #             )
    #     return joint_positions

    def get_skrobot(self, attachments=None):
        attachments = attachments or []

        currj = self.getj()
        self.robot_model.angle_vector(currj)

        link_list = self.robot_model.link_list.copy()
        joint_list = self.robot_model.joint_list.copy()
        for i, attachment in enumerate(attachments):
            position, quaternion = attachment.grasp_pose
            link = skrobot.model.Link(
                parent=self.robot_model.tipLink,
                pos=position,
                rot=geometry.quaternion_matrix(quaternion)[:3, :3],
                name=f"attachment_link{i}",
            )
            joint = skrobot.model.FixedJoint(
                child_link=link,
                parent_link=self.robot_model.tipLink,
                name=f"attachment_joint{i}",
            )
            link_list.append(link)
            joint_list.append(joint)
        return skrobot.model.RobotModel(
            link_list=link_list,
            joint_list=joint_list,
            root_link=self.robot_model.root_link,
        )

    def planj(
        self,
        targj,
        obstacles=None,
        attachments=None,
        self_collisions=True,
        **kwargs,
    ):
        obstacles = [] if obstacles is None else obstacles
        attachments = [] if attachments is None else attachments
        with pybullet_planning.LockRenderer():
            with pybullet_planning.WorldSaver():
                path = pybullet_planning.plan_joint_motion(
                    body=self.robot,
                    joints=self.joints,
                    end_conf=targj,
                    obstacles=obstacles,
                    attachments=attachments,
                    self_collisions=self_collisions,
                    **kwargs,
                )
        return path

    def grasp(self):
        c = geometry.Coordinate(
            *pybullet_planning.get_link_pose(self.robot, self.ee)
        )
        while not self.gripper.detect_contact():
            c.translate([0, 0, 0.001])
            j = self.solve_ik(c.pose, rotation_axis="z")
            if j is None:
                return
            yield from self.movej(j)
        self.gripper.activate()

    def ungrasp(self):
        self.gripper.release()
