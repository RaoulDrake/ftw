from acme.wrappers import observation_action_reward
from acme import specs
import dm_env


def make_individual_environment_spec_from_marl_env(environment: dm_env.Environment) -> specs.EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment.

    This implementation for MARL environments assumes that specs for observation, action & reward
    are represented as a tuple.

    This happens automatically when you convert an OpenAI Gym type of MARL environment to a
    dm_env.Environment via ftw.wrappers.MARLGymWrapper if your implementation meets
    the following requirements:
        -   implements the following attributes/properties:
            -   'observation_space'
            -   'action_space'
            -   'reward_space'
        -   all these attributes/properties are gym.spaces.Tuple,

    This function only works as expected when the following conditions hold for every tuple:
        -   each spec tuple has as many elements as agents in the environment
        -   the elements in a tuple are all identical,
            i.e. the specifications are the same for every agent.
    """
    observation_spec = environment.observation_spec()
    if isinstance(observation_spec, observation_action_reward.OAR):
        observation_spec = observation_action_reward.OAR(
            observation=observation_spec.observation[0],
            action=observation_spec.action[0],
            reward=observation_spec.reward[0],
        )
    else:
        observation_spec = observation_spec[0]  # .replace(shape=observation_spec.shape[0])

    action_spec = environment.action_spec()[0]

    reward_spec = environment.reward_spec()[0]

    discount_spec = environment.discount_spec()
    if len(discount_spec.shape) > 1:
        discount_spec = discount_spec[0]

    return specs.EnvironmentSpec(
        observations=observation_spec,
        actions=action_spec,
        rewards=reward_spec,
        discounts=discount_spec)
