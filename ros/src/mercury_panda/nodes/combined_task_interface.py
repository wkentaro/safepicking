#!/usr/bin/env python

import IPython

import rospy

from reorientbot_task_interface import ReorientbotTaskInterface
from safepicking_task_interface import SafepickingTaskInterface


class CombinedTaskInterface(
    SafepickingTaskInterface, ReorientbotTaskInterface
):
    def run(self):
        SafepickingTaskInterface.run(self, place=False)

        ee_to_world = self.pi.get_pose("tipLink")

        ReorientbotTaskInterface.run(self, target=ee_to_world[0])


rospy.init_node("combined_task_interface")
self = CombinedTaskInterface()
IPython.embed()
