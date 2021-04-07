import numpy as np
import pybullet as p
import pybullet_planning

from .. import geometry


class SuctionGripper:
    def __init__(self, body, link):
        self.body = body
        self.link = link

        self.activated = False
        self.contact_constraint = None

    def activate(self):
        if self.activated:
            return

        self.activated = True

        points = p.getContactPoints(bodyA=self.body, linkIndexA=self.link)
        if points:
            if len(points) > 1:
                print(f"Warning: contact points size is >1: {len(points)}")

            # Handle contact between suction with a rigid object.
            point = points[-1]

            obj_id = point[2]
            contact_link = point[4]
            contact_distance = point[8]

            # in world coordinates
            point_on_ee = point[5]
            point_on_obj = point[6]

            ee_to_world = pybullet_planning.get_link_pose(self.body, self.link)
            world_to_ee = pybullet_planning.invert(ee_to_world)
            T_world_to_ee = geometry.transformation_matrix(*world_to_ee)

            # in ee coordinates
            point_on_ee = geometry.transform_points(
                [point_on_ee], T_world_to_ee
            )[0]
            point_on_obj = geometry.transform_points(
                [point_on_obj], T_world_to_ee
            )[0]

            v_ee_to_obj = (point_on_obj - point_on_ee) * np.sign(
                contact_distance
            )
            v_ee_to_obj /= np.linalg.norm(v_ee_to_obj)

            angle = np.abs(np.arccos(np.dot(v_ee_to_obj, [0, 0, 1])))

            mass = p.getDynamicsInfo(obj_id, -1)[0]
            if mass > 0:
                if angle > np.deg2rad(20):
                    print(
                        "Warning: failed to grasp with surface angle >15 deg: "
                        f"{np.rad2deg(angle):.1f} deg"
                    )
                    return

                # simulate compliance of suction gripper
                T_obj_to_obj_af_in_ee = geometry.transformation_matrix(
                    point_on_ee - point_on_obj,
                    geometry.quaternion_from_vec2vec(v_ee_to_obj, [0, 0, 1]),
                )
                T_obj_to_obj_af_in_ee = geometry.transform_around(
                    T_obj_to_obj_af_in_ee, point_on_obj
                )
                T_obj_to_world = geometry.transformation_matrix(
                    *pybullet_planning.get_pose(obj_id)
                )
                T_obj_to_ee = T_world_to_ee @ T_obj_to_world
                T_obj_af_to_ee = T_obj_to_obj_af_in_ee @ T_obj_to_ee
                T_obj_af_to_world = (
                    np.linalg.inv(T_world_to_ee) @ T_obj_af_to_ee
                )

                body_to_world = p.getLinkState(self.body, self.link)[:2]
                if 0:  # w/o compliance
                    obj_to_world = p.getBasePositionAndOrientation(obj_id)
                else:  # w/ compliance
                    obj_to_world = geometry.pose_from_matrix(T_obj_af_to_world)
                world_to_body = pybullet_planning.invert(body_to_world)
                obj_to_body = pybullet_planning.multiply(
                    world_to_body, obj_to_world
                )
                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.body,
                    parentLinkIndex=self.link,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0),
                )
                p.changeConstraint(self.contact_constraint, maxForce=10)

    def step_simulation(self):
        # this function must be called after p.stepSimulation()
        if self.grasped_object is not None:
            points = p.getClosestPoints(
                bodyA=self.grasped_object,
                linkIndexA=-1,
                bodyB=self.body,
                linkIndexB=self.link,
                distance=0.01,
            )
            if not points:
                # surface is apart more than 1cm
                print("Warning: dropping grasped object as surfaces are apart")
                self.release()

        # FIXME: force on contraint is noisy
        # if self.contact_constraint is None:
        #     return
        # force = p.getConstraintState(self.contact_constraint)
        # if force[2] <= -20:
        #     print(
        #         "Warning: dropping grasped object as force_z <= -20N:",
        #         force[2],
        #     )
        #     self.release()

    def release(self):
        if not self.activated:
            return

        self.activated = False

        # Release gripped rigid object (if any).
        if self.contact_constraint is not None:
            try:
                p.removeConstraint(self.contact_constraint)
                self.contact_constraint = None
            except Exception:
                pass

    def detect_contact(self):
        body, link = self.body, self.link
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except Exception:
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        return self.grasped_object is not None

    @property
    def grasped_object(self):
        grasped_object = None
        if self.contact_constraint is not None:
            grasped_object = p.getConstraintInfo(self.contact_constraint)[2]
        return grasped_object
