from typing import Optional, Union, Tuple, Dict, Mapping

from ftw import datasets
from ftw.adders import reverb as ftw_adders
from ftw.tf import hyperparameters as hp
from ftw.tf import internal_reward

from acme import specs
import reverb


def create_reverb_tables(batch_size: int,
                         max_queue_size: int,
                         use_pixel_control: bool = False,
                         use_reward_prediction: bool = False,
                         max_pixel_control_buffer_size: int = 100,
                         max_reward_pred_buffer_size: int = 800):
    """Creates the reverb table(s) required by the FTW agent.

    Args:
        batch_size: Batch size used in training.
        max_queue_size: Maximum capacity of queue.
        use_pixel_control: Whether to create a table for the Pixel control auxiliary task.
        use_reward_prediction: Whether to create a table for the Reward prediction auxiliary task.
        max_pixel_control_buffer_size: Maximum capacity of Pixel control replay buffer.
        max_reward_pred_buffer_size: Maximum capacity of each Reward prediction replay buffer
            (one buffer for zero rewards, one for non-zero rewards).

    Returns:
        A triple (tables, can_sample_queue, can_sample_auxiliary), where
        tables is a list containing all created tables, can_sample_queue is a function that returns a bool
        indicating whether a batch of training data can be sampled from the queue (used in the calculation of the
        main losses), and can_sample_auxiliary is a function that returns a bool indicating whether a batch of
        training data can be sampled from the auxiliary replay buffer(s).
    """
    tables = []

    queue = reverb.Table.queue(
        name='queue', max_size=max_queue_size)
    can_sample_queue = lambda: queue.can_sample(batch_size)
    tables.append(queue)

    can_sample_pixel_control = lambda: False
    can_sample_reward_prediction = lambda: False

    if use_pixel_control:
        pixel_control_replay = reverb.Table(
            name='uniform_replay_buffer_pixel_control',
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_pixel_control_buffer_size,
            rate_limiter=reverb.rate_limiters.MinSize(batch_size))
        can_sample_pixel_control = lambda: pixel_control_replay.can_sample(batch_size)
        tables.append(pixel_control_replay)

    if use_reward_prediction:
        nonzero_reward_prediction_replay = reverb.Table(
            name='replay_buffer_nonzero_reward_prediction',
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_reward_pred_buffer_size,
            rate_limiter=reverb.rate_limiters.MinSize(batch_size // 2))
        zero_reward_prediction_replay = reverb.Table(
            name='replay_buffer_zero_reward_prediction',
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_reward_pred_buffer_size,
            rate_limiter=reverb.rate_limiters.MinSize(batch_size // 2))
        can_sample_reward_prediction = lambda: (
                nonzero_reward_prediction_replay.can_sample(batch_size // 2) and
                nonzero_reward_prediction_replay.can_sample(batch_size // 2))
        tables += [nonzero_reward_prediction_replay, zero_reward_prediction_replay]

    can_sample_auxiliary = lambda: False
    if use_pixel_control and use_reward_prediction:
        can_sample_auxiliary = lambda: (
                can_sample_pixel_control() and can_sample_reward_prediction())
    elif use_pixel_control:
        can_sample_auxiliary = can_sample_pixel_control
    elif use_reward_prediction:
        can_sample_auxiliary = can_sample_reward_prediction

    return tables, can_sample_queue, can_sample_auxiliary


def create_datasets(learner_client: reverb.TFClient,
                    environment_spec: specs.EnvironmentSpec,
                    batch_size: int,
                    sequence_length: int,
                    extra_spec: Optional[Dict] = None,
                    use_pixel_control: bool = False,
                    use_reward_prediction: bool = False,
                    reward_prediction_sequence_length: int = 3):
    """Creates the dataset(s) required by the FTW agent.

    Args:
        learner_client: A reverb.TFClient connected to the reverb server holding the required reverb tables.
        environment_spec: An acme.specs.EnvironmentSpec namedtuple containing the specs of the environment.
        batch_size: Batch size used in training.
        sequence_length: Length of unroll sequences used in training (for calculation of main losses and
            Pixel control auxiliary loss).
        extra_spec: A dictionary containing extra specs required for training, such as logits or core state.
        use_pixel_control: Whether to create a dataset for the Pixel control auxiliary task.
        use_reward_prediction: Whether to create a dataset for the Reward prediction auxiliary task.
        reward_prediction_sequence_length: Length of reward prediction sequences.
            Defaults to 3, as in the FTW and UNREAL agents

    Returns:
        A 4-element tuple of tf.Dataset objects for each respective task
        (where queue is used in the calculation of the main losses):
            (queue_dataset, pixel_control_dataset, nonzero_reward_prediction_dataset, zero_reward_prediction_dataset)
    """
    queue_dataset = datasets.make_reverb_rnn_sequence_fifo_sampler_dataset(
        client=learner_client,
        table='queue',
        environment_spec=environment_spec,
        batch_size=batch_size,
        extra_spec=extra_spec,
        sequence_length=sequence_length,
        prefetch_size=1)

    pixel_control_dataset = None
    if use_pixel_control:
        pixel_control_dataset = datasets.make_reverb_rnn_sequence_fifo_sampler_dataset(
            client=learner_client,
            table='uniform_replay_buffer_pixel_control',
            environment_spec=environment_spec,
            batch_size=batch_size,
            extra_spec=extra_spec,
            sequence_length=sequence_length,
            prefetch_size=1)

    nonzero_reward_prediction_dataset = None
    zero_reward_prediction_dataset = None
    if use_reward_prediction:
        nonzero_reward_prediction_dataset = datasets.make_reverb_fifo_sampler_dataset(
            client=learner_client,
            table='replay_buffer_nonzero_reward_prediction',
            environment_spec=environment_spec,
            batch_size=batch_size // 2,
            extra_spec={},
            sequence_length=reward_prediction_sequence_length,
            prefetch_size=1)
        zero_reward_prediction_dataset = datasets.make_reverb_fifo_sampler_dataset(
            client=learner_client,
            table='replay_buffer_zero_reward_prediction',
            environment_spec=environment_spec,
            batch_size=batch_size // 2,
            extra_spec={},
            sequence_length=reward_prediction_sequence_length,
            prefetch_size=1)

    return (queue_dataset, pixel_control_dataset,
            nonzero_reward_prediction_dataset, zero_reward_prediction_dataset)


def create_adders(server_address: str,
                  sequence_length: int,
                  use_pixel_control: bool = False,
                  use_reward_prediction: bool = False,
                  reward_prediction_sequence_length: int = 3,
                  reward_prediction_sequence_period: int = 1,
                  pad_end_of_episode: bool = False,
                  delta_encoded: bool = True):
    """Creates the reverb adders required by the FTW actor.

    Args:
        server_address: Address of the reverb server responsible for storing training data.
        sequence_length: Length of unroll sequences used in training (for calculation of main losses and
            Pixel control auxiliary loss).
        use_pixel_control: Whether to create an adder for the Pixel control auxiliary task.
        use_reward_prediction: Whether to create an adder for the Reward prediction auxiliary task.
        reward_prediction_sequence_length: Length of reward prediction sequences.
            Defaults to 3, as in the FTW and UNREAL agents
        reward_prediction_sequence_period: Period with which to add Reward prediction sequences to the respective
            replay buffer. Defaults to 1, i.e. at every step, the last reward_prediction_sequence_length steps are
            added to the replay buffer.
        pad_end_of_episode: Whether to pad sequences with zero-like steps at the the end of an episode, if necessary.
            Defaults to False.
        delta_encoded: Whether to use compression for the adder. May lower RAM requirements.
            See documentation of dm-acme's adders for more details. Defaults to True.

    Returns:
        A tuple (adder, rp_adder), where adder is the main adder and rp_adder is either None
        (if use_reward_prediction=False) or the adder required for the Reward prediction auxiliary task.
    """
    # Component(s) to add things into replay.
    adder_priority_fns = {'queue': lambda x: 1.0}
    if use_pixel_control:
        adder_priority_fns['uniform_replay_buffer_pixel_control'] = lambda x: 1.0
    adder = ftw_adders.NonOverlappingRNNSequenceAdder(
        client=reverb.Client(server_address),
        sequence_length=sequence_length,
        priority_fns=adder_priority_fns,
        delta_encoded=delta_encoded,
        pad_end_of_episode=pad_end_of_episode
    )
    rp_adder = None
    if use_reward_prediction:
        rp_adder = ftw_adders.MultiSequenceAdder(
            client=reverb.Client(server_address),
            sequence_lengths={
                'replay_buffer_nonzero_reward_prediction': reward_prediction_sequence_length,
                'replay_buffer_zero_reward_prediction': reward_prediction_sequence_length},
            periods={
                'replay_buffer_nonzero_reward_prediction': reward_prediction_sequence_period,
                'replay_buffer_zero_reward_prediction': reward_prediction_sequence_period},
            priority_fns={
                'replay_buffer_nonzero_reward_prediction': lambda x: 1.0,
                'replay_buffer_zero_reward_prediction': lambda x: 1.0},
            should_insert_fns={
                'replay_buffer_nonzero_reward_prediction': lambda x: x[-1].reward != 0.,
                'replay_buffer_zero_reward_prediction': lambda x: x[-1].reward == 0.},
            delta_encoded=delta_encoded,
            pad_end_of_episode=pad_end_of_episode)

    return adder, rp_adder


def initialize_hypers(
        slow_core_period_min_max: Tuple[int, int] = (5, 20),
        slow_core_period_init_value: Optional[int] = None,
        learning_rate: Union[float, Tuple[float, float]] = (1e-5, 5 * 1e-3),
        entropy_cost: Union[float, Tuple[float, float]] = (5 * 1e-4, 1e-2),
        reward_prediction_cost: Union[float, Tuple[float, float]] = (0.1, 1.0),
        pixel_control_cost: Union[float, Tuple[float, float]] = (0.01, 0.1),
        kld_prior_fixed_cost: Union[float, Tuple[float, float]] = (1e-4, 0.1),
        kld_prior_posterior_cost: Union[float, Tuple[float, float]] = (1e-3, 1.0),
        scale_grads_fast_to_slow: Union[float, Tuple[float, float]] = (0.1, 1.0),
) -> Mapping[str, hp.Hyperparameter]:
    """Create and initialize all hyperparameters required by the FTW agent.

    All arguments can either be supplied as a 2-tuple (min, max), indicating a range to be used in the
    random initialization of the corresponding hyperparameter, or as a scalar value, in which case the corresponding
    hyperparameter will be initialized with this exact value.
    Please note, however, that if a scalar value is used to initialize slow_core_period to an exact value,
    calling perturb() on the resulting hyperparameter will have no effect.

    Args:
        slow_core_period_min_max: (Inclusive) lower and upper bound for random initialization
            of the period used for the slow core of the RPTH module.
            See docstring for RPTH module (in ftw.tf.networks.recurrence) for more details.
        slow_core_period_init_value: Optional. If not None, the period used for the slow core
            of the RPTH module will be initialized with this exact value, instead of being
            initialized randomly.
            See docstring for RPTH module (in ftw.tf.networks.recurrence) for more details.
        learning_rate: Learning rate used in training.
        entropy_cost: Multiplier for the entropy loss.
        reward_prediction_cost: Multiplier for the Reward prediction loss.
        pixel_control_cost: Multiplier for the Pixel control loss.
        kld_prior_fixed_cost: Multiplier for the Kullback-Leibler divergence loss between
            a fixed Multivariate Normal Diagonal (MVNDiag) distribution and
            the prior (MVNDiag) distribution as produced by the RPTH module's slow core.
        kld_prior_posterior_cost: Multiplier for the Kullback-Leibler divergence loss between
            the prior (MVNDiag) distribution as produced by the RPTH module's slow core and
            the posterior (MVNDiag) distribution as produced by the RPTH module's fast core.
        scale_grads_fast_to_slow: Scaling factor for the gradients flowing from fast to slow core of the RPTH module.

    Returns:
        A dictionary containing all created hyperparameters.
        Keys of this dictionary correspond to the argument names of this function,
        except for the key 'slow_core_period', which results from the argument
        slow_core_period_min_max (and possibly slow_core_init_value).
    """
    slow_core_period = hp.IntHyperparameter(min_value=slow_core_period_min_max[0],
                                            max_value=slow_core_period_min_max[1],
                                            initial_value=slow_core_period_init_value,
                                            name='slow_core_period')
    learning_rate = hp.FloatHyperparameter(init_value_or_range=learning_rate,
                                           name='learning_rate')
    entropy_cost = hp.FloatHyperparameter(init_value_or_range=entropy_cost,
                                          name='entropy_cost')
    reward_prediction_cost = hp.FloatHyperparameter(init_value_or_range=reward_prediction_cost,
                                                    name='reward_prediction_cost')
    pixel_control_cost = hp.FloatHyperparameter(init_value_or_range=pixel_control_cost,
                                                name='pixel_control_cost')
    kld_prior_fixed_cost = hp.FloatHyperparameter(init_value_or_range=kld_prior_fixed_cost,
                                                  name='kld_prior_fixed_cost')
    kld_prior_posterior_cost = hp.FloatHyperparameter(init_value_or_range=kld_prior_posterior_cost,
                                                      name='kld_prior_posterior_cost')
    scale_grads_fast_to_slow = hp.FloatHyperparameter(init_value_or_range=scale_grads_fast_to_slow,
                                                      name='scale_grads_fast_to_slow')
    return {
        'period': slow_core_period,
        'learning_rate': learning_rate,
        'entropy_cost': entropy_cost,
        'reward_prediction_cost': reward_prediction_cost,
        'pixel_control_cost': pixel_control_cost,
        'kld_prior_fixed_cost': kld_prior_fixed_cost,
        'kld_prior_posterior_cost': kld_prior_posterior_cost,
        'scale_grads_fast_to_slow': scale_grads_fast_to_slow
    }


def initialize_internal_rewards(
        num_events: int = 1,
        init_value_or_range: Union[float, Tuple[float, float]] = (0.1, 1.0),
) -> internal_reward.InternalRewards:
    """Creates and initializes the internal rewards required by the FTW agent.

    Args:
        num_events: Number of events returned by the environment.
        init_value_or_range: A scalar value of type int for initialization with an exact value,
            or a tuple (min, max), where min, max are of type int for initialization by drawing
            a sample from a log-uniform distribution defined over (min, max).

    Returns:
        An InternalRewards object.
    """
    return internal_reward.InternalRewards(num_events=num_events,
                                           init_value_or_range=init_value_or_range)
