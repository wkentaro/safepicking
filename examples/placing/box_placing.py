#!/usr/bin/env python

import numpy as np
import pybullet_planning as pp

import mercury

import _placing


class Env(_placing.Env):
    def get_place_waypoints(self, place_pose):
        waypoints = []
        c = mercury.geometry.Coordinate(*place_pose)
        if self._waypoints == "annotation_norot":
            c.translate([0, -0.02, 0], wrt="world")
            waypoints.append(c.pose)
            c.translate([0, 0.02, 0.02], wrt="world")
            waypoints.append(c.pose)
            c.translate([0, 0.02, 0.03], wrt="world")
            waypoints.append(c.pose)
        elif self._waypoints == "annotation":
            c.translate([0, -0.02, 0], wrt="world")
            waypoints.append(c.pose)
            c.translate([0, 0.02, 0.02], wrt="world")
            waypoints.append(c.pose)
            c.translate([0, 0, 0.03], wrt="world")
            c.rotate([np.deg2rad(30), 0, 0], wrt="world")
            waypoints.append(c.pose)
        else:
            assert self._waypoints == "none"
            waypoints.append(c.pose)
        c.translate([0, 0, 0.2], wrt="world")
        waypoints.append(c.pose)
        return waypoints[::-1]

    def _init_objects(self):
        objects = {}

        # bin
        obj = mercury.pybullet.create_bin(
            X=0.335, Y=0.31, Z=0.11, color=(0.7, 0.7, 0.7, 0.5)
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.5105999999999966,
                    -0.004099999999999998,
                    0.08820000000000094,
                ),
                (0.0, 0.0, -0.003999989333334819, 0.9999920000106667),
            ),
        )
        pp.draw_aabb(pp.get_aabb(obj), color=(0, 0, 0, 1))
        objects["bin"] = obj

        parent = obj

        # cracker_box
        class_id = 2
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(1, 0, 0, 1),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    -0.03843769431114197,
                    -0.06841924041509628,
                    -0.020100004971027374,
                ),
                (
                    0.6950822472572327,
                    0.003310413332656026,
                    0.7187591791152954,
                    0.01532843615859747,
                ),
            ),
            parent=parent,
        )
        objects["cracker_box"] = obj

        # sugar_box
        class_id = 3
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(0, 1, 0, 1),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    -0.059823607206344604,
                    0.06029653549194336,
                    -0.031200002878904343,
                ),
                (
                    0.6950822472572327,
                    0.003310413332656026,
                    0.7187591791152954,
                    0.01532843615859747,
                ),
            ),
            parent=parent,
        )
        objects["sugar_box"] = obj

        # mustard_bottle
        class_id = 5
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=0.1,
            rgba_color=(0, 0, 1, 1),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.1017511785030365,
                    -0.06474190950393677,
                    -0.026600003242492676,
                ),
                (
                    0.16755934059619904,
                    0.695159912109375,
                    0.6874131560325623,
                    -0.12704220414161682,
                ),
            ),
            parent=parent,
        )
        objects["mustard_bottle"] = obj

        # tomato_can
        class_id = 4
        obj = mercury.pybullet.create_mesh_body(
            visual_file=True,
            collision_file=mercury.pybullet.get_collision_file(
                mercury.datasets.ycb.get_visual_file(class_id)
            ),
            mass=mercury.datasets.ycb.masses[class_id],
            rgba_color=(1, 1, 0, 1),
        )
        mercury.pybullet.set_pose(
            obj,
            (
                (
                    0.059504538774490356,
                    0.07194063067436218,
                    -0.02120000123977661,
                ),
                (
                    -0.03331788629293442,
                    0.7073509097099304,
                    0.7032650113105774,
                    0.06295282393693924,
                ),
            ),
            parent=parent,
        )
        objects["tomato_can"] = obj

        for name, obj in objects.items():
            pp.set_dynamics(obj, lateralFriction=0.7)

        self.objects = objects


if __name__ == "__main__":
    waypoints = "annotation"
    pose_errors = [
        # no error
        ([0, 0, 0], [0, 0, 0, 0]),
        # collide with the cracker_box
        ([0, 0.02, 0], [0, 0, 0, 0]),
        # collide with the bin
        ([0, -0.02, 0], [0, 0, 0, 0]),
    ]
    for i, pose_error in enumerate(pose_errors):
        env = Env(
            # mp4=f"{waypoints}-pose_error{i}.mp4",
            waypoints=waypoints,
        )
        env.run(
            target_obj=env.objects["tomato_can"],
            pose_error=pose_error,
        )
        pp.disconnect()
