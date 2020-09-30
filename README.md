# For The Win (FTW) Agent implementation.

**[Overview](#overview)** | **[Installation](#installation)** |
**[Documentation](https://ftw.readthedocs.io/en/0.1.8/)** | **[Examples](examples)**

This repository has the following goals:
    
- implement the neural network module and 
- the population-based training framework

introduced by the 2019 paper "Human-level performance in 3D multiplayer games with 
population-based reinforcement learning" by Jaderberg et al. 
(available at https://science.sciencemag.org/content/364/6443/859).

The implementation is based on [TensorFlow](https://www.tensorflow.org/) 2, 
and makes use of the [dm-sonnet](https://github.com/deepmind/sonnet), 
[dm-acme](https://github.com/deepmind/acme) and 
[dm-reverb](https://github.com/deepmind/reverb) libraries offered by Deepmind.

## Overview
This repository offers the following:

- The following neural network modules based on the FTW paper
    
    - Visual embedding (Convolutional neural network)
    - DNC memory (taken from [dnc](https://github.com/deepmind/dnc) and modified to 
    be compatible to TensorFlow 2)
    - Variational Unit
    - Recurrent processing with temporal hierarchy
    - Auxiliary task modules: Pixel control & Reward prediction
    
- FtwNetwork, RNNPixelControlNetwork & RewardPredictionNetwork classes 
 that can be used to combine the above modules into an end-to-end network suitable for 
 the FTW agent
    
- Replay Buffers for Pixel control & Reward prediction auxiliary tasks 
(adders and datasets for use with [dm-acme](https://github.com/deepmind/acme) and 
[dm-reverb](https://github.com/deepmind/reverb))

- Agent, Actor & Learner classes for the FTW agent

- Support for multi-agent environments (under certain constraints, see 
[Documentation](https://ftw.readthedocs.io/en/0.1.8/))

- Hyperparameter and Internal Rewards classes for population-based training.

- Arena & Chief classes for population-based training in multi-agent environments 
(see [FTW paper](https://science.sciencemag.org/content/364/6443/859) for more details)

- A FTWJobPool class that can be used to spawn multiple threads of FTW learners and 
Arena instances.


However, this repository is still work-in-progress. As such, the following features are not 
supported/implemented yet:

- In the current state, population-based training is not yet fully implemented: 
The Chief class responsible for the evolution of the agent population is still work in progress. 
Therefore, while it can already be used in the training of a population of agents, it does not 
actually execute any evolutionary methods, such as mutation of hyperparameters.
- Currently, the policy module does not support decomposed action spaces such as the one featured in 
the FTW paper.
- Similarly, the Pixel control module does not support decomposed action spaces at the moment.

These features will be added in the near future.

## Installation
Currently, only Linux based OSes are supported (due to dm-reverb).

1. Clone/download the repository.
2. Go to the repository folder
        
        cd ftw/
        
3. It is highly recommended to use a 
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html) 
to manage your dependencies in order to avoid version conflicts:

        python3 -m venv ftw
        source ftw/bin/activate
        pip install --upgrade pip setuptools
        
4. Install with (mind the dot at the end!)

        pip install -e .
        
5. If you want to use the multi-agent examples, you'll need to additionally install 
[ma-gym](https://github.com/koulanurag/ma-gym). Warning: do not install ma-gym from 
github, since the version requirements will install older versions of libraries that 
are incompatible with our version requirements. However, if you have followed the steps 
so far and install ma-gym from the third_party module in this repository, the examples 
will work just fine. If you are in the ftw root directory, type

        cd ftw/third_party/ma_gym
        pip install -e .
        
Again, mind the dot at the end!    
That's it. You're all set!

## Documentation
For more specific information, please visit the 
[Documentation](https://ftw.readthedocs.io/en/0.1.8/).

## Examples
Further examples can be found [here](ftw/examples).
