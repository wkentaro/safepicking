import time

import numpy as np
import path
import pybullet as p
import pybullet_planning

from .. import geometry
from .suction_gripper import SuctionGripper


here = path.Path(__file__).abspath().parent


class PandaRobotInterface:
    def __init__(self):
        self.robot = pybullet_planning.load_pybullet(
            here / "assets/franka_panda/panda_suction.urdf", fixed_base=True
        )
        self.ee = pybullet_planning.link_from_name(self.robot, "tipLink")

        self.gripper = SuctionGripper(
            self.robot, self.ee, graspable_objects=[]
        )

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.robot)
        joints = [p.getJointInfo(self.robot, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.homej = [0, -np.pi / 4, 0, -np.pi / 2, 0, np.pi / 4, np.pi / 4]
        for joint in self.joints:
            p.resetJointState(self.robot, joint, self.homej[joint])

    def setj(self, joint_positions):
        for joint, joint_position in enumerate(joint_positions):
            p.resetJointState(self.robot, joint, joint_position)

    def movej(self, targj, speed=0.01, timeout=5):
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

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
            p.stepSimulation()
            time.sleep(1 / 240)
        print(f"Warning: movej exceeded {timeout} second timeout. Skipping.")
        return False

    def movep(self, pose, speed=0.01):
        targj = self.solve_ik(pose)
        return self.movej(targj, speed=speed)

    def _solve_ik_pybullet(self, pose):
        n_joints = p.getNumJoints(self.robot)
        lower_limits = []
        upper_limits = []
        for i in range(n_joints):
            joint_info = p.getJointInfo(self.robot, i)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.ee,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            restPoses=self.homej,
            maxNumIterations=1000,
            residualThreshold=1e-5,
        )
        joint_positions = [joint_positions[i] for i in self.joints]
        return joint_positions

    def solve_ik(self, pose):
        with pybullet_planning.WorldSaver(), pybullet_planning.LockRenderer():
            targj = pybullet_planning.inverse_kinematics(
                self.robot,
                self.ee,
                pose,
                custom_limits={
                    9: pybullet_planning.UNBOUNDED_LIMITS,
                    10: pybullet_planning.UNBOUNDED_LIMITS,
                },
            )
        if targj is None:
            raise RuntimeError("Failed to find an IK solution")
        return [targj[j] for j in self.joints]

    def planj(
        self, targj, obstacles=None, attachments=None, self_collisions=True
    ):
        obstacles = [] if obstacles is None else obstacles
        attachments = [] if attachments is None else attachments
        with pybullet_planning.WorldSaver(), pybullet_planning.LockRenderer():
            path = pybullet_planning.plan_joint_motion(
                body=self.robot,
                joints=self.joints,
                end_conf=targj,
                obstacles=obstacles,
                attachments=attachments,
                self_collisions=self_collisions,
            )
        return path

    def planp(
        self, pose, obstacles=None, attachments=None, self_collisions=True
    ):
        targj = self.solve_ik(pose)
        return self.planj(
            targj=targj,
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=self_collisions,
        )

    def grasp(self):
        c = geometry.Coordinate(
            *pybullet_planning.get_link_pose(self.robot, self.ee)
        )
        while not self.gripper.detect_contact():
            c.translate([0, 0, 0.001])
            self.movep(c.pose)
        self.gripper.activate()

    def ungrasp(self):
        self.gripper.release()
