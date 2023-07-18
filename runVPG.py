# import matplotlib.pyplot as plt
import matplotlib
from absl import flags
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
import gpu
import logging
from acme.jax.experiments import config as configClass
import os
import vpg
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad.context
os.environ["MUJOCO_GL"]="egl"
os.environ["EGL_DEVICE_ID"] = "1"
matplotlib.use("TkAgg")
terminals = ['gnome-terminal', 'gnome-terminal-tabs', 'xterm',
      'tmux_session', 'current_terminal', 'output_to_files']
import gym
for i in gym.envs.registry.all():
  print(i.id)

contGymEnvs = ["BipedalWalker-v3","CarRacing-v2","MountainCarContinuous-v0","Pendulum-v1", "Swimmer-v4"]
discGymEnvs = ["CartPole-v4", "FrozenLake-v1"]
gymEnv = contGymEnvs[3]
# gymEnv = discGymEnvs[1]

runDistributed = True
envName = 'gym:'+gymEnv
seed=0
numSteps = int(1e8)
evalEvery = 50000
evalEpisodes = 10
tensorboardDir = gymEnv + '_vpg_expt'


def build_experiment_config():

    # This preliminary env stuff seems to be fine as is
    suite, task = envName.split(':', 1)

    tempEnv = helpers.make_environment(suite, task)
    if "Bounded" in str(type(tempEnv.action_spec())):
        envType = "continuous"
        print("Continuous action spec")
    elif "Discrete" in str(type(tempEnv.action_spec())):
        envType = "discrete"
        print("Discrete action spec")
    else:
        raise Exception("Cannot determine action spec")
    del tempEnv

    env_factory = lambda _: helpers.make_environment(suite, task)

    layer_sizes = (256, 256)
    network_factory = lambda spec: vpg.make_networks(spec, layer_sizes)

    # Configure and construct the agent builder.
    config = vpg.VPGConfig(unroll_length=8,#For pendulum, 200 is the max ep length
                           obs_normalization_fns_factory=vpg.build_mean_std_normalizer,
                           entropy_cost=3e-2,
                           normalize_advantage=True,
                           normalize_value=True,
                           batch_size=192
                           )

    agent_builder = vpg.VPGBuilder(config)


    #I think the rest is fine as is
    def make_logger(label, steps_key='learner_steps',i=0):
        from acme.utils.loggers.terminal import TerminalLogger
        from acme.utils.loggers.tf_summary import TFSummaryLogger
        from acme.utils.loggers import base, aggregators
        summaryDir = "tensorboardOutput/"+tensorboardDir
        terminal_logger = TerminalLogger(label=label, print_fn=logging.info)
        tb_logger = TFSummaryLogger(summaryDir, label=label, steps_key=steps_key)
        serialize_fn = base.to_numpy
        logger = aggregators.Dispatcher([terminal_logger, tb_logger], serialize_fn)
        return logger

    checkpointingConfig = configClass.CheckpointingConfig()
    checkpointingConfig.max_to_keep = 5
    checkpointingConfig.directory = '/home/kenny/acme/' + task + '/VPG'
    checkpointingConfig.time_delta_minutes = 10
    checkpointingConfig.add_uid = False

    return experiments.ExperimentConfig(
        builder=agent_builder,
        environment_factory=env_factory,
        network_factory=network_factory,
        seed=seed,
        max_num_actor_steps=numSteps,
        logger_factory=make_logger,
        checkpointing=checkpointingConfig
    )


def main(_):

  config = build_experiment_config()

  if runDistributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=16,
    )
    resources = {
        # The 'actor' and 'evaluator' keys refer to
        # the Launchpad resource groups created by Program.group()
        'actor':
            PythonProcess(  # Dataclass used to specify env vars and args (flags) passed to the Python process
                env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                         XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'evaluator':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'counter':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'replay':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="1",
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'learner':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES="1",
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }

    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
                       launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
                       terminal=terminals[1], local_resources=resources)
    worker.wait()
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=evalEvery,
        num_eval_episodes=evalEpisodes)

if __name__ == '__main__':
    gpu.SetGPU(-3,True)
    app.run(main)
    # Render - maybe also render once per epoch?