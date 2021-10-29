#!/usr/bin/env python

import IPython
import pybullet_planning as pp

import rospy

from base_task_interface import BaseTaskInterface
from reorientbot_task_interface import ReorientbotTaskInterface
from safepicking_task_interface import SafepickingTaskInterface


class CombinedTaskInterface:
    def __init__(self, base):
        self.base = base
        self.safepicking = SafepickingTaskInterface(self.base)
        self.reorientbot = ReorientbotTaskInterface(self.base)

    def run(self):
        self.safepicking.run(self, place=False)

        if self.base._env.fg_object_id is None:
            target = self.base.pi.get_pose("tipLink")[0]
        else:
            target = pp.get_pose(self.base._env.fg_object_id)[0]

        self.reorientbot.run(self, target=target)


rospy.init_node("combined_task_interface")
base = BaseTaskInterface
self = CombinedTaskInterface(base)
IPython.embed()
