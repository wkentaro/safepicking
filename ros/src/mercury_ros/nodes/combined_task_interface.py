#!/usr/bin/env python

import IPython
import pybullet_planning as pp

import rospy

from reorientbot_task_interface import ReorientbotTaskInterface
from safepicking_task_interface import SafepickingTaskInterface


class CombinedTaskInterface(
    SafepickingTaskInterface, ReorientbotTaskInterface
):
    def run(self):
        SafepickingTaskInterface.run(self, place=False)

        if self._env.fg_object_id is None:
            target = self.pi.get_pose("tipLink")[0]
        else:
            target = pp.get_pose(self._env.fg_object_id)[0]

        ReorientbotTaskInterface.run(self, target=target)


rospy.init_node("combined_task_interface")
self = CombinedTaskInterface()
IPython.embed()
