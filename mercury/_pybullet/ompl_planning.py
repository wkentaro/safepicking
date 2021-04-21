import os.path as osp
import sys

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

here = path.Path(__file__).abspath().parent
sys.path.insert(0, osp.join(here, "../../src/ompl/py-bindings"))
from ompl import base as ob  # NOQA
from ompl import geometric as og  # NOQA
from ompl import util as ou  # NOQA


class pbValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, ri, obstacles=None, min_distances=None):
        super().__init__(si)
        self.ri = ri
        self.ndof = len(self.ri.joints)
        self.obstacles = obstacles or []
        self.min_distances = min_distances or {}
        self.lower, self.upper = ri.get_bounds()
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)

    def isValid(self, state):
        if not self.check_joint_limits(state):
            return False

        with pp.LockRenderer(), pp.WorldSaver():
            self.ri.setj(state)
            for attachment in self.ri.attachments:
                attachment.assign()
            return self.check_self_collision() and self.check_collision(
                self.obstacles
            )

    def check_self_collision(self):
        # TODO(wkentaro)
        return True

    def check_collision(self, ids_to_check):
        if len(ids_to_check) == 0:
            return True

        is_colliding = False
        for link in pp.get_links(self.ri.robot):
            min_distance = self.min_distances.get((self.ri.robot, link), 0)
            is_colliding |= (
                len(
                    p.getClosestPoints(
                        bodyA=self.ri.robot,
                        bodyB=ids_to_check[-1],
                        linkIndexA=link,
                        linkIndexB=-1,
                        distance=min_distance,
                    )
                )
                > 0
            )

        for attachment in self.ri.attachments:
            min_distance = self.min_distances.get((attachment.child, -1), 0)
            is_colliding |= (
                len(
                    p.getClosestPoints(
                        attachment.child,
                        ids_to_check[-1],
                        distance=min_distance,
                    )
                )
                > 0
            )

        if is_colliding:
            return False
        else:
            return self.check_collision(ids_to_check[0:-1])

    def check_joint_limits(self, state):
        for i in range(self.ndof):
            if state[i] > self.upper[i] or state[i] < self.lower[i]:
                return False
        return True

    def sample_state(self):
        q = (
            np.random.random(self.ndof) * (self.upper - self.lower)
            + self.lower
        )
        if self.isValid(q):
            return q
        else:
            return self.sample_state()


class PbPlanner:
    def __init__(self, ri, obstacles=None, min_distances=None):
        ndof = len(ri.joints)

        lower, upper = ri.get_bounds()
        bounds = ob.RealVectorBounds(ndof)
        for i in range(ndof):
            bounds.setLow(i, lower[i])
            bounds.setHigh(i, upper[i])

        self.space = ob.RealVectorStateSpace(ndof)
        self.space.setBounds(bounds)

        self.si = ob.SpaceInformation(self.space)

        self.validityChecker = pbValidityChecker(
            self.si, ri=ri, obstacles=obstacles, min_distances=min_distances
        )
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()

    def plan(self, start_q, goal_q):
        # start and goal configs
        start = ob.State(self.space)
        for i in range(len(start_q)):
            start[i] = start_q[i]

        goal = ob.State(self.space)
        for i in range(len(start_q)):
            goal[i] = goal_q[i]

        # setup and solve
        pdef = ob.ProblemDefinition(self.si)
        pdef.setStartAndGoalStates(start, goal)
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(self.si)
        )
        optimizingPlanner = og.RRTConnect(self.si)
        optimizingPlanner.setRange(0.1)
        optimizingPlanner.setProblemDefinition(pdef)
        optimizingPlanner.setup()
        solved = optimizingPlanner.solve(solveTime=5)

        if solved:
            path = pdef.getSolutionPath()
            simplifier = og.PathSimplifier(self.si)
            simplifier.simplifyMax(path)
            return path
        else:
            logger.warning("No solution found")
            return None
