# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Blocks domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np


_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)


SUITE = containers.TaggedTasks()


def make_model(n_boxes, xml_file='blocks.xml'):
    """Returns a tuple containing the model XML string and a dict of assets."""
    xml_string = common.read_model(xml_file)
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Remove unused boxes
    for b in range(n_boxes, 4):
        box = xml_tools.find_element(mjcf, 'body', 'box' + str(b))
        box.getparent().remove(box)

    return etree.tostring(mjcf, pretty_print=True), common.ASSETS


@SUITE.add('easy')
def push_1(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
           environment_kwargs=None):
    """Returns block env with one block"""
    n_boxes = 1
    physics = Physics.from_xml_string(
        *make_model(n_boxes=n_boxes, xml_file='blocks.xml'))
    task = Blocks(n_boxes=n_boxes,
                  fully_observable=fully_observable,
                  random=random,
                  include_target=False,
                  easy_init=True
                  )
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
        **environment_kwargs)

@SUITE.add('hard')
def stack_2(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
           environment_kwargs=None):
    """Returns block env with one block"""
    n_boxes = 2
    physics = Physics.from_xml_string(
        *make_model(n_boxes=n_boxes, xml_file='blocks.xml'))
    task = Blocks(n_boxes=n_boxes,
                  fully_observable=fully_observable,
                  random=random,
                  include_target=False,
                  easy_init=True
                  )
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def bounded_joint_pos(self, joint_names):
        """Returns joint positions as (sin, cos) values."""
        joint_pos = self.named.data.qpos[joint_names]
        return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.named.data.qvel[joint_names]

    def body_2d_pose(self, body_names, orientation=True):
        """Returns positions and/or orientations of bodies."""
        if not isinstance(body_names, str):
            # Broadcast indices.
            body_names = np.array(body_names).reshape(-1, 1)
        pos = self.named.data.xpos[body_names, ['x', 'z']]
        if orientation:
            ori = self.named.data.xquat[body_names, ['qw', 'qy']]
            return np.hstack([pos, ori])
        else:
            return pos

    def touch(self):
        return np.log1p(self.data.sensordata)

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(
            self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)


class Blocks(base.Task):
    """A Stack `Task`: stack the boxes."""

    def __init__(self, n_boxes, fully_observable, random=None, include_target=True, easy_init=False):
        """Initialize an instance of the `Stack` task.

        Args:
          n_boxes: An `int`, number of boxes to stack.
          fully_observable: A `bool`, whether the observation should contain the
            positions and velocities of the boxes and the location of the target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._n_boxes = n_boxes
        self._box_names = ['box' + str(b) for b in range(n_boxes)]
        self._box_joint_names = []
        self.easy_init = easy_init
        for name in self._box_names:
            for dim in 'xyz':
                self._box_joint_names.append('_'.join([name, dim]))
        self._fully_observable = fully_observable

        super(Blocks, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Local aliases
        randint = self.random.randint
        uniform = self.random.uniform
        model = physics.named.model
        data = physics.named.data

        # Find a collision-free random initial configuration.
        penetrating = True
        i = 0
        while penetrating:
            i += 1

            # Randomise box locations.
            if self.easy_init:
                for name in self._box_names:
                    data.qpos[name + '_x'] = uniform(-.3, .3)
                    data.qpos[name +
                              '_z'] = physics.named.model.geom_size['box0', 0]
                    #data.qpos[name + '_y'] = uniform(0, 2*np.pi)
            else:
                for name in self._box_names:
                    data.qpos[name + '_x'] = uniform(.1, .3)
                    data.qpos[name + '_z'] = uniform(0, .7)
                    data.qpos[name + '_y'] = uniform(0, 2*np.pi)

            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 4 if self.easy_init else physics.data.ncon > 0

        super(Blocks, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()

        if self._fully_observable:
            obs['box_pos'] = physics.body_2d_pose(self._box_names)
            obs['box_vel'] = physics.joint_vel(self._box_joint_names)

        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return None
