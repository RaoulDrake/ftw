import multiprocessing
from concurrent import futures

from absl import app
from absl import flags
import acme
from ftw.agents.tf.ftw.agent import FTW
from acme import wrappers
from ftw import wrappers as ftw_wrappers
from ftw.pbt.arena import MARLEnvironmentLoop

from acme import specs

import gym
import ma_gym

import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

flags.DEFINE_string('level', 'PongDuel-v0', 'Which Atari level to play.')

# Training Settings.
flags.DEFINE_integer('population_size', 2, 'Number of agents in the population.')
flags.DEFINE_integer('episodes', 450000, 'Number of episodes per generation.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 20, 'Unroll length in agent steps.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

FLAGS = flags.FLAGS

tfd = tfp.distributions


def main(_):
    tf.random.set_seed(FLAGS.seed)

    env_fn = lambda env_name: wrappers.wrap_all(
        environment=gym.make(env_name),
        wrappers=[
            ftw_wrappers.MarlGymWrapper,
            ftw_wrappers.ObservationActionRewardMarlWrapper,
            wrappers.SinglePrecisionWrapper
        ]
    )

    agents = []

    env = env_fn(FLAGS.level)

    for i in range(FLAGS.population_size):
        env_spec = acme.make_environment_spec(env)
        env_spec = specs.EnvironmentSpec(
            observations=env_spec.observations[i],
            actions=env_spec.actions[i],
            rewards=env_spec.rewards[i],
            discounts=env_spec.discounts
        )

        state_embedding = snt.nets.MLP(output_sizes=[128, 128])

        agent = FTW(
            environment_spec=env_spec,
            sequence_length=FLAGS.unroll_length,
            embed=state_embedding,
            use_pixel_cotrol=False,
            uint_pixels_to_float=False,
            core_type='rpth',
            agent_id=i
        )
        agents.append(agent)

    env_loops = []
    for i in range(len(agents) // FLAGS.population_size):
        env_loop = MARLEnvironmentLoop(
            environment=env_fn(FLAGS.level),
            actors=[agents[j] for j in range(len(agents))],
            label=f'environment_loop_{i}'
        )
        env_loops.append(env_loop)

    with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Start the environment loops and mark each future with its environment loop / agent index.
        env_loops_future_to_idx = {
            executor.submit(
                lambda: env_l.run(num_episodes=FLAGS.episodes)): j
            for j, env_l in enumerate(env_loops)}

        env_loops_is_done = [False for _ in env_loops_future_to_idx]
        for future in futures.as_completed(env_loops_future_to_idx):
            idx = env_loops_future_to_idx[future]
            try:
                env_loops_is_done[idx] = future.result()
            except Exception as exc:
                print(f'Environment Loop {idx} threw an exception: {exc}')

        print('Training complete.')


if __name__ == '__main__':
    app.run(main)
