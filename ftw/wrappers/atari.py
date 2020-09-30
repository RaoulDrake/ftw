import functools

import dm_env
from acme import wrappers
from ftw.wrappers import util_wrappers
import gym


def make_canonical_atari_environment(
        environment_name: str,
        to_float=True,
        evaluation: bool = False) -> dm_env.Environment:
    # TODO: docstring
    env = gym.make(environment_name, full_action_space=True)

    max_episode_len = 108_000 if evaluation else 50_000

    return wrappers.wrap_all(env, [
        wrappers.GymAtariAdapter,  # adds lives count to observation => observation=(rgb, l_count)
        functools.partial(
            wrappers.AtariWrapper,
            to_float=to_float,
            max_episode_len=max_episode_len,
            zero_discount_on_life_loss=True,
        ),
        wrappers.observation_action_reward.ObservationActionRewardWrapper,
        wrappers.SinglePrecisionWrapper
    ])


def make_rgb_atari_environment_wo_frame_stack(
        environment_name: str,
        to_float=True,
        evaluation: bool = False,
        is_non_atari: bool = False,
        full_action_space: bool = True) -> dm_env.Environment:
    # TODO: docstring
    if is_non_atari:
        env = gym.make(environment_name)
    else:
        env = gym.make(environment_name, full_action_space=full_action_space)

    max_episode_len = 108_000 if evaluation else 50_000

    return wrappers.wrap_all(env, [
        wrappers.GymAtariAdapter,  # adds lives count to observation => observation=(rgb, l_count)
        functools.partial(
            wrappers.AtariWrapper,
            to_float=to_float,
            max_episode_len=max_episode_len,
            zero_discount_on_life_loss=True,
            num_stacked_frames=1,
            grayscaling=False
        ),
        util_wrappers.ReshapeOneDimFrameStackWrapper,
        wrappers.observation_action_reward.ObservationActionRewardWrapper,
        wrappers.SinglePrecisionWrapper
    ])
