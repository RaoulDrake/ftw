TensorFlow Modules & Networks
=============================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   ftw.tf.networks.dnc

Auxiliary Task Modules & Networks
---------------------------------

``Pixel Control module``
************************
.. autoclass:: ftw.tf.networks.auxiliary.PixelControl
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``RNN Pixel Control network``
*****************************
.. autoclass:: ftw.tf.networks.auxiliary.RNNPixelControlNetwork
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Reward Prediction module``
****************************
.. autoclass:: ftw.tf.networks.auxiliary.RewardPrediction
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Reward Prediction network``
****************************
.. autoclass:: ftw.tf.networks.auxiliary.RewardPredictionNetwork
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

Distributional modules
----------------------

``Multivariate Normal Diagonal Distribution Head``
**************************************************
.. autoclass:: ftw.tf.networks.distributional.MultivariateNormalDiagHead
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Multivariate Normal Diagonal Distribution Loc Scale Head``
************************************************************
.. autoclass:: ftw.tf.networks.distributional.MultivariateNormalDiagLocScaleHead
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

Embedding Modules
-----------------

``Observation Action Reward Embedding Module``
**********************************************
.. autoclass:: ftw.tf.networks.embedding.OAREmbedding
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

For The Win (FTW) Network
-------------------------

.. automodule:: ftw.tf.networks.ftw_network
   :members:
   :undoc-members:
   :show-inheritance:

Policy Value Head Module
------------------------

``Policy Value Head Module``
****************************
.. autoclass:: ftw.tf.networks.policy_value.PolicyValueHead
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

Recurrent Core Modules
----------------------

``DNC Wrapper Module``
**********************
.. autoclass:: ftw.tf.networks.recurrence.DNCWrapper
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Variational Unit Module``
***************************
.. autoclass:: ftw.tf.networks.recurrence.VariationalUnit
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Periodic Variational Unit Module``
************************************
.. autoclass:: ftw.tf.networks.recurrence.PeriodicVariationalUnit
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``RPTH (Recurrent Processing with Temporal Hierarchy) Module``
**************************************************************
.. autoclass:: ftw.tf.networks.recurrence.RPTH
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Convenience Wrapper for RPTH Module``
***************************************
.. autoclass:: ftw.tf.networks.recurrence.RPTH
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

``Named Tuples for Recurrent Outputs and States``
*************************************************
.. autoclass:: ftw.tf.networks.recurrence.LocScaleDistributionParameters
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ftw.tf.networks.recurrence.PeriodicRNNState
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ftw.tf.networks.recurrence.RPTHState
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ftw.tf.networks.recurrence.RPTHOutput
    :members:
    :undoc-members:
    :show-inheritance:

Vision (Visual Embedding) Modules
---------------------------------

.. autoclass:: ftw.tf.networks.vision.FtwTorso
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
