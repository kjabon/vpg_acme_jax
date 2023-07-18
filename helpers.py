# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for rl_continuous experiments."""


from gym import wrappers as gymWrappers
import my_gym_wrapper
from time import time

import functools
import os

from absl import flags
from acme import specs
from acme import wrappers
import my_atari_wrapper
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
import atari_py  # pylint:disable=unused-import
import dm_env
import gym
import haiku as hk
import jax.numpy as jnp


_VALID_TASK_SUITES = ('gym', 'control')


FLAGS = flags.FLAGS
def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = True,
    oar_wrapper: bool = False,
    num_stacked_frames: int = 4,
    flatten_frame_stack: bool = False,
    grayscaling: bool = True,
) -> dm_env.Environment:
  """Loads the Atari environment."""
# Internal logic.
  version = 'v0' if sticky_actions else 'v4'
  level_name = f'{level}NoFrameskip-{version}'
  env = gym.make(level_name, full_action_space=False)
  env = gymWrappers.RecordVideo(env, './videos/' + level + '/' + str(time()) + '/')

  if not "ram" in level:
      wrapper_list = [
          wrappers.GymAtariAdapter,
          functools.partial(
              wrappers.AtariWrapper,
              to_float=True,
              max_episode_len=108_000,
              num_stacked_frames=num_stacked_frames,
              flatten_frame_stack=flatten_frame_stack,
              grayscaling=grayscaling,
              zero_discount_on_life_loss=zero_discount_on_life_loss,
          ),
          wrappers.SinglePrecisionWrapper,
      ]
  else:
      wrapper_list = [
          wrappers.GymAtariAdapter,
          functools.partial(
              my_atari_wrapper.AtariWrapperFromRam,
              max_episode_len=108_000,
              num_stacked_frames=num_stacked_frames,
              zero_discount_on_life_loss=zero_discount_on_life_loss,
              flatten_frame_stack=flatten_frame_stack,
              to_float =True
          ),
          wrappers.SinglePrecisionWrapper,
      ]
  if oar_wrapper:
    # E.g. IMPALA and R2D2 use this particular variant.
    wrapper_list.append(wrappers.ObservationActionRewardWrapper)

  return wrappers.wrap_all(env, wrapper_list)



def make_dqn_atari_network_from_pixels(
    environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  def pixelNetwork(inputs):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  return make_dqn_atari_network_helper(environment_spec, pixelNetwork)

def make_dqn_atari_network_from_ram(
      environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  def ramNetwork(inputs):
    inputs = jnp.array(inputs,dtype=jnp.float32)
    if inputs.ndim == 1:
        if inputs.shape[0] == 1:
            inputs = jnp.reshape(inputs, (1, -1))
        else:
            inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
    model = hk.Sequential([
        hk.nets.MLP([512, 512],activate_final=True),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  return make_dqn_atari_network_helper(environment_spec, ramNetwork)


def make_dqn_atari_network_helper(
    environment_spec: specs.EnvironmentSpec,
    network_factory) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  network_hk = hk.without_apply_rng(hk.transform(network_factory)) #I.e., without apply rng means a deterministic policy

  obsSpace = environment_spec.observations
  if isinstance(obsSpace, specs.DiscreteArray):
      dummy_obs = jnp.zeros((1,))
  else: dummy_obs = utils.zeros_like(obsSpace)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs), apply=network_hk.apply)
  typed_network = networks_lib.non_stochastic_network_to_typed(network)
  return dqn.DQNNetworks(policy_network=typed_network)

def make_environment(suite: str, task: str) -> dm_env.Environment:
  """Makes the requested continuous control environment.

  Args:
    suite: One of 'gym' or 'control'.
    task: Task to load. If `suite` is 'control', the task must be formatted as
      f'{domain_name}:{task_name}'

  Returns:
    An environment satisfying the dm_env interface expected by Acme agents.
  """

  if suite not in _VALID_TASK_SUITES:
    raise ValueError(
        f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

  if suite == 'gym':
    env = gym.make(task)
    # env = gymWrappers.RecordVideo(env, './videos/' + task + '/' + str(time()) + '/')

    # Make sure the environment obeys the dm_env.Environment interface.
    env = my_gym_wrapper.GymWrapper(env)

  elif suite == 'control':
    # Load dm_suite lazily not require Mujoco license when not using it.
    from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
    domain_name, task_name = task.split(':')
    env = dm_suite.load(domain_name, task_name)
    env = wrappers.ConcatObservationWrapper(env)

  # Wrap the environment so the expected continuous action spec is [-1, 1].
  # Note: this is a no-op on 'control' tasks.
  # env = wrappers.CanonicalSpecWrapper(env, clip=True)
  env = wrappers.SinglePrecisionWrapper(env)
  return env
