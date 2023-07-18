from typing import Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders

from vpg import config as vpg_config
from vpg import learning
from vpg import networks as vpg_networks
from vpg import normalization

from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import numpy as np
import optax
import reverb


class VPGBuilder(
    builders.ActorLearnerBuilder[vpg_networks.VPGNetworks,
                                 actor_core_lib.FeedForwardPolicyWithExtra,
                                 reverb.ReplaySample]):
  """VPG Builder."""

  def __init__(
      self,
      config: vpg_config.VPGConfig,
  ):
    """Creates VPG builder."""
    self._config = config

    # An extra step is used for bootstrapping when computing advantages.
    self._sequence_length = config.unroll_length + 1

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: actor_core_lib.FeedForwardPolicyWithExtra,
  ) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    del policy
    extra_spec = {
        'log_prob': np.ones(shape=(), dtype=np.float32),
    }
    signature = adders_reverb.SequenceAdder.signature(
        environment_spec, extra_spec, sequence_length=self._sequence_length)
    return [
        reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=self._config.batch_size,
            signature=signature)
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset.
    The iterator batch size is computed as follows:
    Let:
      B := learner batch size (config.batch_size)
      H := number of hosts (jax.process_count())
      D := number of local devices per host
    The Reverb iterator will load batches of size B // (H * D). After wrapping
    the iterator with utils.multi_device_put, this will result in an iterable
    that provides B // H samples per item, with B // (H * D) samples placed on
    each local device. In a multi-host setup, each host has its own learner
    node and builds its own instance of the iterator. This will result
    in a total batch size of H * (B // H) == B being consumed per learner
    step (since the learner is pmapped across all devices). Note that
    jax.device_count() returns the total number of devices across hosts,
    i.e. H * D.
    Args:
      replay_client: the reverb replay client
    Returns:
      A replay buffer iterator to be used by the local devices.
    """
    iterator_batch_size, ragged = divmod(self._config.batch_size,
                                         jax.device_count())
    if ragged:
      raise ValueError(
          'Learner batch size must be divisible by total number of devices!')

    # We don't use datasets.make_reverb_dataset() here to avoid interleaving
    # and prefetching, that doesn't work well with can_sample() check on update.
    # NOTE: Value for max_in_flight_samples_per_worker comes from a
    # recommendation here: https://git.io/JYzXB
    max_in_flight_samples_per_worker = int(2 * self._config.batch_size /
                                          jax.process_count())

    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=self._config.replay_table_name,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker)
    dataset = dataset.batch(iterator_batch_size, drop_remainder=True)
    dataset = dataset.as_numpy_iterator()
    return utils.multi_device_put(iterable=dataset, devices=jax.local_devices())

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[actor_core_lib.FeedForwardPolicyWithExtra],
  ) -> Optional[adders.Adder]:
    """Creates an adder which handles observations."""
    del environment_spec, policy
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return adders_reverb.SequenceAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        period=self._sequence_length - 1,
        sequence_length=self._sequence_length,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: vpg_networks.VPGNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del replay_client

    # if callable(self._config.learning_rate):
    #   optimizer = optax.chain(
    #       # optax.clip_by_global_norm(self._config.max_gradient_norm),
    #       optax.scale_by_adam(eps=self._config.adam_epsilon),
    #       optax.scale_by_schedule(self._config.learning_rate), optax.scale(-1))
    # else:
    optimizer = optax.chain(
      optax.clip_by_global_norm(self._config.max_gradient_norm),
      optax.scale_by_adam(eps=self._config.adam_epsilon),
      optax.scale(-self._config.learning_rate))

    obs_normalization_fns = None
    if self._config.obs_normalization_fns_factory is not None:
      obs_normalization_fns = self._config.obs_normalization_fns_factory(
          environment_spec.observations)

    return learning.VPGLearner(
        vpg_networks=networks,
        iterator=dataset,
        discount=self._config.discount,
        # value_cost=self._config.value_cost,
        # ppo_clipping_epsilon=self._config.ppo_clipping_epsilon,
        normalize_advantage=self._config.normalize_advantage,
        normalize_value=self._config.normalize_value,
        # normalization_ema_tau=self._config.normalization_ema_tau,
        # clip_value=self._config.clip_value,
        # value_clipping_epsilon=self._config.value_clipping_epsilon,
        gae_lambda=self._config.gae_lambda,
        counter=counter,
        random_key=random_key,
        optimizer=optimizer,
        num_epochs=self._config.num_epochs,
        num_minibatches=self._config.num_minibatches,
        logger=logger_fn('learner'),
        # log_global_norm_metrics=self._config.log_global_norm_metrics,
        # metrics_logging_period=self._config.metrics_logging_period,
        pmap_axis_name=self._config.pmap_axis_name,
        obs_normalization_fns=obs_normalization_fns,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: actor_core_lib.FeedForwardPolicyWithExtra,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    assert variable_source is not None
    policy_variable_client = variable_utils.VariableClient(
        variable_source,
        'params',
        device='cpu',
        update_period=self._config.variable_update_period)
    actor = actor_core_lib.batched_feed_forward_with_extras_to_actor_core(
        policy)
    actor = actors.GenericActor(
        actor, random_key, policy_variable_client, adder, backend='cpu')

    if self._config.obs_normalization_fns_factory is not None:
      obs_normalization_fns = self._config.obs_normalization_fns_factory(
          environment_spec.observations)
      obs_norm_variable_client = variable_utils.VariableClient(
          variable_source,
          'obs_normalization_params',
          device='cpu',
          update_period=self._config.variable_update_period)
      actor = normalization.NormalizationActorWrapper(
          actor, obs_normalization_fns, obs_norm_variable_client, backend='cpu')
    return actor

  def make_policy(
      self,
      networks: vpg_networks.VPGNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False) -> actor_core_lib.FeedForwardPolicyWithExtra:
    del environment_spec
    return vpg_networks.make_inference_fn(networks, evaluation)
