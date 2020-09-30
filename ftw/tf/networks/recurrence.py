from typing import Optional, Union, Sequence, Callable
from ftw.types import Initializer

import functools
import collections

from acme.tf import utils as tf2_utils
from ftw.tf.networks.dnc import access, dnc
from ftw.tf.networks import distributional as ftw_distributional
from ftw.tf import hyperparameters as hp

import numpy as np
import tree
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FnTensorToTensorOrDistribution = Callable[[tf.Tensor], Union[tf.Tensor, tfd.Distribution]]


class DNCWrapper(snt.RNNCore):
    """DNC Memory Wrapper module wrapping an LSTM controller, a DNC MemoryAccess module and an output layer together.

    This module implements the DNC memory introduced by Deepmind by connecting an LSTM controller, a DNC MemoryAccess
    module and an output layer, which are all provided to the module via constructor arguments.
    In contrast to the original TensorFlow version 1.x implementation of the DNC module at
    https://github.com/deepmind/dnc, this module lets the user supply controller, memory and output modules as
    constructor arguments, instead of accepting parameters for the creation of these in the constructor arguments and
    then building the modules during construction (as is done in the original implementation).
    """

    def __init__(self,
                 lstm: snt.LSTM,
                 memory: access.MemoryAccess,
                 output_layer: Optional[Union[snt.Module, FnTensorToTensorOrDistribution]] = None,
                 clip_value=None,
                 name: str = 'dnc_wrapper'):
        """Initializes the DNCWrapper module

        The clip_value Args info was taken from the original TensorFlow version 1.x implementation of the DNC module
        at https://github.com/deepmind/dnc.

        Args:
            lstm: LSTM module of type sonnet.LSTM.
            memory: DNC MemoryAccess module.
            output_layer: Output layer that outputs either a tf.Tensor or a tfp.Distribution.
            clip_value: clips controller and core output values to between [-clip_value, clip_value]` if specified.
            name: Name for the module.
        """
        super().__init__(name=name)

        self._controller = lstm
        self._access = memory

        self._access_output_size = np.prod(self._access.output_size.as_list())
        self._clip_value = clip_value or 0

        self._output_layer = output_layer

    def _clip_if_enabled(self, x):
        """Clips the value(s) of a tensor by clip_value, if clipping is enabled.

        Args:
            x: A tf.Tensor.

        Returns:
            x: A (possibly clipped) tf.Tensor.
        """
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        else:
            return x

    def __call__(self, inputs, prev_state):
        """DNC core.

        The Args and Returns info of this docstring was taken from the original TensorFlow version 1.x implementation
        of the DNC module at https://github.com/deepmind/dnc.

        Args:
          inputs: Tensor input.
          prev_state: A `DNCState` tuple containing the fields `access_output`,
              `access_state` and `controller_state`. `access_state` is a 3-D Tensor
              of shape `[batch_size, num_reads, word_size]` containing read words.
              `access_state` is a tuple of the access module's state, and
              `controller_state` is a tuple of controller module's state.

        Returns:
          A tuple `(output, next_state)` where `output` is a tensor and `next_state`
          is a `DNCState` tuple containing the fields `access_output`,
          `access_state`, and `controller_state`.
        """

        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        batch_flatten = snt.Flatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

        controller_output, controller_state = self._controller(
            controller_input, prev_controller_state)

        controller_output = self._clip_if_enabled(controller_output)
        controller_state = tree.map_structure(self._clip_if_enabled, controller_state)

        access_output, access_state = self._access(controller_output,
                                                   prev_access_state)

        output = tf.concat([controller_output, batch_flatten(access_output)], 1)
        if self._output_layer is not None:
            output = self._output_layer(output)
            if isinstance(output, tf.Tensor):
                output = self._clip_if_enabled(output)

        return output, dnc.DNCState(
            access_output=access_output,
            access_state=access_state,
            controller_state=controller_state)

    def initial_state(self, batch_size: int, **unused_kwargs):
        """Returns the initial DNCState namedtuple, containing all elements of the recurrent state.

        Elements of the DNCState recurrent state are:
            - controller_state: state of the controller module (LSTM)
            - access_state: state of the DNC MemoryAccess module
            - access_output: last output of the DNC MemoryAccess module.

        Returns:
            A DNCState namedtuple.
        """
        return dnc.DNCState(
            controller_state=self._controller.initial_state(batch_size),
            access_state=self._access.initial_state(batch_size),
            access_output=tf.zeros(
                [batch_size] + self._access.output_size.as_list(), dtype=tf.float32))


class VariationalUnit(snt.RNNCore):
    """Variational Unit module as introduced by the FTW paper.

    Can be used with a shared DNC MemoryAccess module, if supplied via constructor arguments.

    See also the FTW paper for more information, especially Figure S11 of the supplementary material.
    """

    def __init__(self,
                 hidden_size: int,
                 num_dimensions: int,
                 shared_memory: Optional[access.MemoryAccess] = None,
                 dnc_clip_value=None,
                 use_dnc_linear_projection: bool = True,
                 init_scale: float = 0.1,
                 min_scale: float = 1e-6,
                 tanh_mean: bool = False,
                 fixed_scale: bool = False,
                 w_init: Optional[Initializer] = None,  # tf.initializers.VarianceScaling(1e-4),
                 b_init: Optional[Initializer] = None,  # tf.initializers.Zeros(),
                 name='variational_unit'):
        """Initialization.

        Args:
          hidden_size: Hidden size of LSTM.
          num_dimensions: Number of dimensions of MVN distribution.
          shared_memory: (Possibly shared) DNC MemoryAccess module. Optional. If None, no memory is used.
          dnc_clip_value: Only used when shared_memory is not None. Clip value used by DNC module for clipping the
            (LSTM) controller output and state, as well as the linear output.
          use_dnc_linear_projection: Only used when shared_memory is not None. Whether the DNC module outputs the
            concatenated LSTM and memory outputs or a linear projection thereof (with the same hidden size as the LSTM).
            In the original DNC, this linear projection is used. Defaults to True.
          init_scale: Initial standard deviation.
          min_scale: Minimum standard deviation.
          tanh_mean: Whether to transform the mean (via tanh) before passing it to the distribution.
          fixed_scale: Whether to use a fixed variance.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
          name: Name for the module.
        """
        super().__init__(name=name)
        use_dnc_linear_projection = use_dnc_linear_projection and (shared_memory is not None)
        dnc_output_layer = None
        if shared_memory is not None:
            if use_dnc_linear_projection:
                dnc_output_layer = snt.Linear(output_size=hidden_size, name='dnc_linear_output')
            self._core = DNCWrapper(
                lstm=snt.LSTM(hidden_size=hidden_size),
                memory=shared_memory,
                output_layer=dnc_output_layer,
                clip_value=dnc_clip_value,
                name='dnc')
        else:
            self._core = snt.LSTM(hidden_size=hidden_size)
        self._distribution = ftw_distributional.MultivariateNormalDiagLocScaleHead(
            num_dimensions=num_dimensions,
            init_scale=init_scale, min_scale=min_scale,
            tanh_mean=tanh_mean,
            fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init)

    def __call__(self, inputs, state):
        """Updates recurrent state and produces mean and scale of a Multivariate Normal Diagonal distribution.

        Args:
            inputs: Tensor input.
            state: Recurrent state. LSTMState namedtuple if LSTM core is used, DNCState namedtuple if DNC core is used.

        Returns:
            A tuple (LocScaleDistributionParameters, core_state),
            where LocScaleDistributionParameters is a namedtuple containing the mean and scale (stddev) of a
            Multivariate Normal Diagonal distribution, and core_state is the updated recurrent state
            (LSTMState namedtuple if LSTM core is used, DNCState namedtuple if DNC core is used).
        """
        core_output, core_state = self._core(inputs, state)
        loc, scale = self._distribution(core_output)
        return LocScaleDistributionParameters(loc=loc, scale=scale), core_state

    def initial_state(self, batch_size: int, **unused_kwargs):
        """Returns the initial recurrent state.

        Recurrent state is an LSTMState namedtuple if LSTM core is used, or DNCState namedtuple if DNC core is used.

        Returns:
            Initial state namedtuple (LSTMState or DNCState, depending on which core is used) of recurrent core.
        """
        return self._core.initial_state(batch_size=batch_size)


PeriodicRNNState = collections.namedtuple('PeriodicRNNState', ['core_state', 'output', 'step'])


class PeriodicVariationalUnit(VariationalUnit):
    """Periodic Variational Unit module as introduced by the FTW paper.

    This module implements a Variational Unit that only updates its hidden state every period steps
    (i.e. if step % period = 0), as used by the FTW agent.

    See also the FTW paper for more information, especially Figure S11 of the supplementary material.
    """

    def __init__(self,
                 period: Union[int, tf.Variable],
                 hidden_size: int,
                 num_dimensions: int,
                 shared_memory: Optional[access.MemoryAccess] = None,
                 dnc_clip_value=None,
                 use_dnc_linear_projection: bool = True,
                 init_scale: float = 0.1,
                 min_scale: float = 1e-6,
                 tanh_mean: bool = False,
                 fixed_scale: bool = False,
                 w_init: Optional[Initializer] = None,  # tf.initializers.VarianceScaling(1e-4),
                 b_init: Optional[Initializer] = None,  # tf.initializers.Zeros(),
                 name='periodic_variational_unit'):
        """Initialization.

        Args:
          period: Periodically update the recurrent core every period steps.
            This module keeps a step counter in its state (which resets to 0 when initial_state() is called).
            The recurrent core of this module only updates its hidden state when step % period == 0.
          For all other arguments, please see the docstrings for
            - VariationalUnit (in ftw.tf.networks.recurrence) and
            - MultivariateNormalDiagLocScaleHead (in ftw.tf.networks.distributional).
        """
        self._period = period
        if isinstance(period, int):
            # Note: Although period is an int, we construct the constant as tf.float32.
            # This is to ensure the tf.math.floormod operation in __call__ works with the
            # step counter in our recurrent state, which is also tf.float32 (to ensure that
            # all of the recurrent state has the same dtype).
            self._period = tf.constant(period, dtype=tf.float32, shape=(),
                                       name='variational_unit_period')
        if isinstance(period, tf.Variable) and period.dtype != tf.float32:
            raise ValueError(f'PeriodicVariationalUnit: period had dtype {period.dtype}, '
                             f'but tf.float32 is required.')
        super().__init__(
            hidden_size=hidden_size,
            num_dimensions=num_dimensions,
            shared_memory=shared_memory,
            dnc_clip_value=dnc_clip_value,
            use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init,
            name=name)
        # Internalize the number of (distribution) dimensions, so we know the sizes/shapes
        # for the initial core_output in self.initial_state().
        self._num_dimensions = num_dimensions

    def __call__(self, inputs, state):
        """(Possibly) updates recurrent state and produces mean and scale (stddev) of a MVNDiag distribution.

        Only updates recurrent state and produces output if step % period = 0, where step is part of the
        PeriodicRNNState namedtuple given by state and period is supplied as an argument at construction.

        Args:
            inputs: Tensor input.
            state: PeriodicRNNState namedtuple containing recurrent core_state, previous output and step counter.

        Returns:
            A tuple (LocScaleDistributionParameters, PeriodicRNNState), where LocScaleDistributionParameters is a
            namedtuple containing mean and scale (stddev) of a Multivariate Normal Diagonal distribution,
            and PeriodicRNNState is a namedtuple containing recurrent core_state, previous output and step counter.

        """
        # Unpack state.
        prev_core_state, prev_output, step = state.core_state, state.output, state.step

        # Compute new output and state.
        new_core_output, new_core_state = self._core(inputs, prev_core_state)

        # Determine whether to update output and state.
        should_update = tf.math.equal(tf.math.floormod(step, self._period), 0)

        # Compute mean (loc) and stddev (scale) for a Multivariate Normal Diagonal distribution.
        loc, scale = self._distribution(new_core_output)

        new_output = LocScaleDistributionParameters(loc=loc, scale=scale)

        # Maybe update state and output.
        core_state = tree.map_structure(
            functools.partial(tf.compat.v1.where, should_update),
            new_core_state, prev_core_state)
        output = tree.map_structure(
            functools.partial(tf.compat.v1.where, should_update),
            new_output, prev_output)

        # Update step counter.
        step += 1.0

        return output, PeriodicRNNState(
            core_state=core_state, output=output, step=step)

    def initial_state(self, batch_size: int, **unused_kwargs):
        """Returns the initial recurrent state.

        Recurrent state is a PeriodicRNNState namedtuple containing recurrent core_state (core_state),
        previous output (output) and step counter (step), where output is a LocScaleDistributionParameters namedtuple
        containing the mean and scale (stddev) of a Multivariate Normal Diagonal distribution.

        Returns:
            Initial recurrent state as a PeriodicRNNState namedtuple.
        """
        return PeriodicRNNState(
            core_state=self._core.initial_state(batch_size=batch_size),
            output=LocScaleDistributionParameters(
                loc=tf.zeros(shape=[batch_size, self._num_dimensions], dtype=tf.float32),
                scale=tf.zeros(shape=[batch_size, self._num_dimensions], dtype=tf.float32)),
            step=tf.zeros(shape=[batch_size], dtype=tf.float32, name='vu_step_counter'))


RPTHState = collections.namedtuple('RPTHState', ['z', 'core_state', 'step'])  # 'distribution_params',
RPTHOutput = collections.namedtuple('RPTHOutput', ['z', 'distribution_params'])
LocScaleDistributionParameters = collections.namedtuple(
    'LocScaleDistributionParameters', ['loc', 'scale'])


class RPTH(snt.RNNCore):
    """Recurrent processing with temporal hierarchy module, as introduced by the FTW paper.

    This module consists of 2 or more Variational Units,
    where one Variational Unit updates its hidden state every step and is responsible for the posterior distribution,
    and the other Variational Unit updates its hidden state only if step % period = 0 and is responsible for the
    prior distribution.
    Optionally, a DNC MemoryAccess module can be supplied as a constructor argument, which will be shared by all cores,
    i.e. all cores write to and read from the same memory, and memory weights are shared among all cores.

    Warning: Please note that while support for more than 2 cores is implemented, it is not tested yet and is thus
    highly discouraged. Please proceed with care if you wish to use this feature.
    """

    def __init__(self,
                 period: Union[int, Sequence[int], tf.Variable, Sequence[tf.Variable]],
                 hidden_size: int = 256,
                 num_dimensions: int = 256,
                 dnc_clip_value=None,
                 use_dnc_linear_projection: bool = True,
                 init_scale: float = 0.1,
                 min_scale: float = 1e-6,
                 tanh_mean: bool = False,
                 fixed_scale: bool = False,
                 use_tfd_independent: bool = False,
                 w_init: Optional[Initializer] = None,  # tf.initializers.VarianceScaling(1e-4),
                 b_init: Optional[Initializer] = None,  # tf.initializers.Zeros(),
                 shared_memory: Optional[access.MemoryAccess] = None,
                 strict_period_order: bool = True,
                 scale_gradients_fast_to_slow: Union[float, Sequence[float], tf.Variable, Sequence[tf.Variable]] = 1.0,
                 name: str = 'rpth'):
        """Initializes the RPTH module.

        Args:
          period: Periodically update the recurrent core(s) every period steps. If period is a
            scalar int value, only one slow core will be used. If period is a sequence of scalar int values,
            multiple slow cores, each with the given period, will be used. Note that when supplying a
            sequence of scalar int values that is not in descending order, it will be sorted automatically,
            unless strict_period_order=False.
          strict_period_order: See period for further information. Defaults to True, i.e. periods will
            automatically be sorted in descending order, if they were not supplied in this order.
          For all other arguments, please see the docstrings for
            - VariationalUnit (in ftw.tf.networks.recurrence) and
            - MultivariateNormalDiagLocScaleHead (in ftw.tf.networks.distributional).
        """
        # TODO: scale_gradients_fast_to_slow doesnt' support sequences yet,
        #  and only scales the gradient from the fast core to all slow cores.
        super().__init__(name=name)
        # Internalize arguments.
        self._hidden_size = hidden_size
        self._num_dimensions = num_dimensions
        self._use_tfd_independent = use_tfd_independent
        self._period = period
        if isinstance(period, int) or isinstance(period, tf.Variable):
            self._period = [period]
        if strict_period_order and sorted(self._period, reverse=True) != self._period:
            print("Warning: 'period' was not sorted in descending order, which is the standard expected "
                  "behaviour of the RPTH module. We will automatically sort 'period' in descending order. "
                  "If you explicitly wish to use the order you passed, use 'strict_period_order=False'")
            self._period = sorted(self._period, reverse=True)
        self._shared_memory = shared_memory
        self._scale_gradients_fast_to_slow = scale_gradients_fast_to_slow

        # Construct Variational Units.
        vu_kwargs = dict(
            hidden_size=hidden_size,
            num_dimensions=num_dimensions,
            shared_memory=shared_memory,
            dnc_clip_value=dnc_clip_value,
            use_dnc_linear_projection=use_dnc_linear_projection,
            init_scale=init_scale,
            min_scale=min_scale,
            tanh_mean=tanh_mean,
            fixed_scale=fixed_scale,
            w_init=w_init, b_init=b_init)
        self._variational_units = []

        # Slow core(s).
        for i, period in enumerate(self._period):
            period_int = period if isinstance(period, int) else int(period.numpy())
            vu_name = f'slow_variational_unit_{i}_period_{period_int}'
            vu = PeriodicVariationalUnit(
                period=period,
                **vu_kwargs,
                name=vu_name)
            self._variational_units.append(vu)

        # Fast core.
        last_variational_unit = VariationalUnit(**vu_kwargs, name='fast_variational_unit')
        self._variational_units.append(last_variational_unit)

        # Set up call method, depending on whether a shared memory is used or not.
        # In this module's __call__ method, we will only invoke self._call_fn and return the results.
        if self._shared_memory is not None:  # Cores are DNC instances.
            self._call_fn = self._call_with_shared_mem
        else:  # Cores are LSTM instances.
            self._call_fn = self._call_without_shared_mem

        # Stored for convenience.
        self._num_cores = len(self._variational_units)

    def __call__(self, inputs, state):
        """"""
        return self._call_fn(inputs, state)

    def _call_without_shared_mem(self, inputs, state):
        prev_core_state, prev_output, step = state.core_state, state.output, state.step

        prev_hidden_states = list(prev_core_state.hidden)  # Previous LSTM hidden states for each core.
        prev_cell_states = list(prev_core_state.cell)  # Previous LSTM cell states for each core.

        new_hidden_states = []
        new_cell_states = []

        # Slow cores.
        new_slow_vu_outputs = []  # Stores mean (loc) and stddev (scale) for each (slow) output distribution.
        for i, vu in enumerate(self._variational_units[:-1]):
            # Gather previous hidden states from every core that is yet to step.
            prev_other_hidden_states = prev_hidden_states[i + 1:]

            # Inputs to a slow core are (LSTM) hidden states from all other cores,
            # including the fast core. Previous (t-1) states are taken from cores that are yet to step,
            # current (t) states from cores that already stepped.
            vu_input = tf2_utils.batch_concat(new_hidden_states + prev_other_hidden_states + new_slow_vu_outputs)
            # State for LSTM includes its own private hidden and cell state.
            vu_state = PeriodicRNNState(
                core_state=snt.LSTMState(hidden=prev_hidden_states[i], cell=prev_cell_states[i]),
                output=prev_output[i],
                step=step)

            # Step through slow core, possibly updating its state.
            new_vu_output, new_vu_state = vu(vu_input, vu_state)

            # Add results to corresponding lists.
            new_slow_vu_outputs.append(new_vu_output)
            new_hidden_states.append(new_vu_state.core_state.hidden)
            new_cell_states.append(new_vu_state.core_state.cell)

        # Fast core.
        # Inputs to fast core are:
        #   - inputs passed to __call__,
        #   - (LSTM) hidden states from all slow cores,
        #   - the mean and stddev of the slow (prior) distribution(s),
        #   - the previous sample (z) of the posterior distribution that the fast core generates.
        prev_z = prev_output[-1]
        # Scale gradients flowing from fast core to slow core(s)
        vu_input_from_slow = tf2_utils.batch_concat(new_hidden_states + new_slow_vu_outputs)
        vu_input_from_slow = snt.scale_gradient(vu_input_from_slow, self._scale_gradients_fast_to_slow)
        vu_input = tf2_utils.batch_concat([inputs, vu_input_from_slow, prev_z])
        vu_state = snt.LSTMState(hidden=prev_hidden_states[-1], cell=prev_cell_states[-1])

        # Step through fast core, updating its state.
        new_vu_output, new_vu_state = self._variational_units[-1](vu_input, vu_state)
        if not self._use_tfd_independent:
            distribution = tfd.MultivariateNormalDiag(loc=new_vu_output.loc, scale_diag=new_vu_output.scale)
        else:
            distribution = tfd.Independent(tfd.Normal(loc=new_vu_output.loc, scale=new_vu_output.scale))
        z = distribution.sample(name='z')

        # Add results to corresponding lists.
        new_hidden_states.append(new_vu_state.hidden)
        new_cell_states.append(new_vu_state.cell)

        # Update step counter.
        step += 1.0

        final_output = RPTHOutput(z=z, distribution_params=tuple(new_slow_vu_outputs + [new_vu_output]))
        final_state = PeriodicRNNState(
            core_state=snt.LSTMState(
                hidden=tuple(new_hidden_states), cell=tuple(new_cell_states)),
            output=tuple(new_slow_vu_outputs + [z]),
            step=step)

        return final_output, final_state

    def _call_with_shared_mem(self, inputs, state):
        prev_core_state, prev_output, step = state.core_state, state.output, state.step

        prev_controller_states = prev_core_state.controller_state  # Previous LSTM states
        prev_hidden_states = [st.hidden for st in prev_controller_states]  # Previous LSTM hidden states.
        prev_access_outputs = list(prev_core_state.access_output)  # Previous MemoryAccess outputs.
        access_state = prev_core_state.access_state  # Previous MemoryAccess state.

        new_controller_states = []
        new_access_outputs = []

        # Slow cores.
        new_slow_vu_outputs = []  # Stores mean (loc) and stddev (scale) for each (slow) output distribution.
        for i, vu in enumerate(self._variational_units[:-1]):
            # Gather previous hidden states and access outputs from every core that is yet to step.
            prev_other_hidden_states = prev_hidden_states[i + 1:]
            prev_other_access_outputs = prev_access_outputs[i + 1:]

            # Inputs to a slow core are hidden states and access outputs from all other cores,
            # including the fast core. Previous (t-1) states and access outputs are taken from cores
            # that are yet to step, current (t) states and access outputs from cores that already stepped.
            vu_input = tf2_utils.batch_concat(
                [st.hidden for st in new_controller_states] + prev_other_hidden_states +
                new_access_outputs + prev_other_access_outputs + new_slow_vu_outputs)
            # State for VariationalUnit includes its own private controller_state and access_output,
            # as well as the shared memory state.
            vu_state = PeriodicRNNState(
                core_state=dnc.DNCState(
                    controller_state=prev_controller_states[i],
                    access_output=prev_access_outputs[i],
                    access_state=access_state),
                output=prev_output[i],
                step=step)

            # Step through slow core, possibly updating its state.
            new_vu_output, new_vu_state = vu(vu_input, vu_state)

            # Update access_state variable, so following cores receive the possibly updated memory state.
            access_state = new_vu_state.core_state.access_state

            # Add results to corresponding lists.
            new_slow_vu_outputs.append(new_vu_output)
            new_controller_states.append(new_vu_state.core_state.controller_state)
            new_access_outputs.append(new_vu_state.core_state.access_output)

        # Fast core.
        # Inputs to fast core are:
        #   - inputs passed to __call__,
        #   - hidden states and access outputs from all slow cores,
        #   - the mean and stddev of the slow (prior) distribution(s),
        #   - the previous sample (z) of the posterior distribution that the fast core generates.
        prev_z = prev_output[-1]
        # Scale gradients flowing from fast core to slow core(s)
        vu_input_from_slow = tf2_utils.batch_concat(
            [st.hidden for st in new_controller_states] + new_access_outputs + new_slow_vu_outputs)
        vu_input_from_slow = snt.scale_gradient(vu_input_from_slow, self._scale_gradients_fast_to_slow)
        vu_input = tf2_utils.batch_concat([inputs, vu_input_from_slow, prev_z])
        vu_state = dnc.DNCState(
                controller_state=prev_controller_states[-1],
                access_output=prev_access_outputs[-1],
                access_state=access_state)

        # Step through fast core, updating its state.
        new_vu_output, new_vu_state = self._variational_units[-1](vu_input, vu_state)
        if not self._use_tfd_independent:
            distribution = tfd.MultivariateNormalDiag(loc=new_vu_output.loc, scale_diag=new_vu_output.scale)
        else:
            distribution = tfd.Independent(tfd.Normal(loc=new_vu_output.loc, scale=new_vu_output.scale))
        z = distribution.sample(name='z')

        # Add results to corresponding lists.
        new_controller_states.append(new_vu_state.controller_state)
        new_access_outputs.append(new_vu_state.access_output)

        # Update access_state variable.
        access_state = new_vu_state.access_state

        # Update step counter.
        step += 1.0

        final_output = RPTHOutput(z=z, distribution_params=tuple(new_slow_vu_outputs + [new_vu_output]))
        final_state = PeriodicRNNState(
            core_state=dnc.DNCState(
                controller_state=tuple(new_controller_states),
                access_output=tuple(new_access_outputs),
                access_state=access_state),
            output=tuple(new_slow_vu_outputs + [z]),
            step=step)

        return final_output, final_state

    def initial_state(self, batch_size: int, **unused_kwargs):
        # Get initial state and step from one of the slow cores.
        init_slow_core_full_state = self._variational_units[0].initial_state(
            batch_size=batch_size)
        init_slow_core_state = init_slow_core_full_state.core_state
        init_slow_output = init_slow_core_full_state.output
        init_step = init_slow_core_full_state.step

        # Create initial core state.
        if isinstance(init_slow_core_state, dnc.DNCState):  # Cores are DNC instances.
            # Create a DNCState (NamedTuple) where
            #   - the 'controller_state' field is a tuple of the LSTM states of all cores,
            #   - the 'access_output' field is a tuple of memory outputs for all cores,
            #   - the 'access_state' field is the (single) state of the shared DNC MemoryAccess module.
            controller_states = tuple([init_slow_core_state.controller_state] * self._num_cores)
            access_outputs = tuple([init_slow_core_state.access_output] * self._num_cores)
            access_state = init_slow_core_state.access_state
            core_state = dnc.DNCState(
                controller_state=controller_states,
                access_output=access_outputs,
                access_state=access_state)
        else:  # Cores are LSTM instances.
            # Create LSTMState (NamedTuple), where the 'hidden' and 'cell' fields are each
            # a tuple of hidden/cell states for all cores.
            hidden_states = tuple([init_slow_core_state.hidden] * self._num_cores)
            cell_states = tuple([init_slow_core_state.cell] * self._num_cores)
            core_state = snt.LSTMState(hidden=hidden_states, cell=cell_states)

        slow_outputs = [init_slow_output] * (self._num_cores - 1)
        z = tf.zeros(shape=(batch_size, self._num_dimensions), dtype=tf.float32)

        return PeriodicRNNState(
            core_state=core_state,
            output=tuple(slow_outputs + [z]),
            step=init_step)


class RPTHZWrapper(snt.RNNCore):
    """Wraps an RPTH core and returns as output only the posterior sample z (instead of an RPTHOutput namedtuple)."""

    def __init__(self,
                 rpth_core: RPTH):
        """Initializes the RPTHZWrapper module.

        Args:
            rpth_core: RPTH recurrent core module (from ftw.tf.networks.recurrence).

        Raises:
            ValueError: If rpth_core is not of type RPTH.
        """
        if not isinstance(rpth_core, RPTH):
            raise ValueError(f"rpth_core must be of type RPTH (from ftw.tf.networks.recurrence) "
                             f"but was of type {type(rpth_core)}.")
        super().__init__(name='rpth_z_wrapper')
        self._core = rpth_core

    def initial_state(self, batch_size: int, **kwargs):
        """Returns the initial recurrent state of the wrapped RPTH module.

        See the docstring for the RPTH module (in ftw.tf.networks.recurrence) for more details.

        Returns:
            Initial recurrent state of the wrapped RPTH module.
        """
        return self._core.initial_state(batch_size, **kwargs)

    def __call__(self, inputs, state):
        """Calls the wrapped RPTH module and returns as output the sample z from the posterior distribution.

        Returns:
            A tuple (z, new_state), where z is the sample of the posterior distribution of the RPTH module output and
            new_state is the updated recurrent state of the RPTH module.
        """
        output, new_state = self._core(inputs, state)

        return output.z, new_state
