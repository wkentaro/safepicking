#!/usr/bin/env python

import pybullet_planning as pp

import mercury

import _placing
import _utils


class Env(_placing.Env):
    def get_place_waypoints(self, place_pose):
        waypoints = []
        c = mercury.geometry.Coordinate(*place_pose)
        if 1:
            if self._target_obj == self.objects["cracker_box_02"]:
                c.translate([0.02, 0, 0], wrt="world")
                waypoints.append(c.pose)

                c.translate([0, 0, 0.02], wrt="world")
                waypoints.append(c.pose)

                c.translate([-0.04, 0, 0.04], wrt="world")
                waypoints.append(c.pose)
            elif self._target_obj == self.objects["mustard_bottle"]:
                c.translate([0, 0, -0.02], wrt="world")
                waypoints.append(c.pose)

                c.translate([0, 0, 0.02], wrt="world")
                waypoints.append(c.pose)

                c.translate([0, 0, 0.04], wrt="world")
                waypoints.append(c.pose)
        else:
            waypoints.append(c.pose)
        c.translate([0, -0.2, 0], wrt="world")
        waypoints.append(c.pose)
        return waypoints[::-1]

    def _init_objects(self):
        objects = {}

        # shelf
        obj = _utils.create_shelf(X=0.29, Y=0.41, Z=0.285, N=2)
        pp.set_color(obj, (0.7, 0.7, 0.7, 0.5))
        mercury.pybullet.set_pose(
            obj,
            (
                (0.3, 0.6476999999999964, 0.02239999999999992),
                (0.0, 0.0, -0.7071067811865478, 0.7071067811865472),
            ),
        )
        objects["shelf"] = obj

        # cracker_box 01
        class_id = 2
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file if self._use_visual else True,
            collision_file=mercury.pybullet.get_collision_file(visual_file),
            mass=0.1,
            # mass=mercury.datasets.ycb.masses[class_id],
            rgba_color=None if self._use_visual else (1, 0, 0),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (0.06400001049041748, 0.16540002822875977, 0.4247000217437744),
                (
                    -5.56090629460045e-16,
                    -5.561200109154029e-16,
                    0.7018561959266663,
                    0.7123186588287354,
                ),
            ),
            parent=objects["shelf"],
        )
        objects["cracker_box_01"] = obj

        # cracker_box 02
        class_id = 2
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file if self._use_visual else True,
            collision_file=mercury.pybullet.get_collision_file(visual_file),
            mass=0.1,
            # mass=mercury.datasets.ycb.masses[class_id],
            rgba_color=None if self._use_visual else (0, 1, 0),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (0.06400001049041748, 0.09970003366470337, 0.4247000217437744),
                (
                    -5.560906823996042e-16,
                    -5.561200638549621e-16,
                    0.7018561959266663,
                    0.7123186588287354,
                ),
            ),
            parent=objects["shelf"],
        )
        objects["cracker_box_02"] = obj

        # mustard_bottle
        class_id = 5
        visual_file = mercury.datasets.ycb.get_visual_file(class_id)
        obj = mercury.pybullet.create_mesh_body(
            visual_file=visual_file if self._use_visual else True,
            collision_file=mercury.pybullet.get_collision_file(visual_file),
            mass=0.1,
            # mass=mercury.datasets.ycb.masses[class_id],
            rgba_color=None if self._use_visual else (0, 0, 1),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (0.0877000093460083, -0.07809999585151672, 0.3904000222682953),
                (0.0, 0.0, -0.5501134395599365, 0.8350899815559387),
            ),
            parent=objects["shelf"],
        )
        objects["mustard_bottle"] = obj

        self.objects = objects


if __name__ == "__main__":
    env = Env(mp4="cracker_box-pose_error03-waypoints.mp4")

    # # collide with the shelf
    # env.run(
    #     env.objects["cracker_box_02"],
    #     pose_error=([0, 0, 0.02], [0, 0, 0, 0]),
    # )

    # # collide with the other box
    # env.run(
    #     env.objects["cracker_box_02"],
    #     pose_error=([-0.02, 0, 0], [0, 0, 0, 0]),
    # )

    # off from the other box
    env.run(
        env.objects["cracker_box_02"],
        pose_error=([0.02, 0, 0], [0, 0, 0, 0]),
    )

    # # collide with the shelf
    # env.run(
    #     env.objects["mustard_bottle"],
    #     pose_error=([0, 0, 0.02], [0, 0, 0, 0]),
    # )
