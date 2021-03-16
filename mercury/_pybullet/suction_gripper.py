import pybullet as p
import pybullet_planning


class SuctionGripper:
    def __init__(self, body, link, graspable_objects):
        self.body = body
        self.link = link
        self.graspable_objects = graspable_objects

        self.activated = False
        self.contact_constraint = None

    def activate(self):
        if self.activated:
            return

        points = p.getContactPoints(bodyA=self.body, linkIndexA=self.link)
        if points:
            # Handle contact between suction with a rigid object.
            for point in points:
                obj_id, contact_link = point[2], point[4]
            if obj_id in self.graspable_objects:
                body_to_world = p.getLinkState(self.body, self.link)[:2]
                obj_to_world = p.getBasePositionAndOrientation(obj_id)
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

            self.activated = True

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
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None
