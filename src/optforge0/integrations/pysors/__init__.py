"""
Those were originally implemented in https://github.com/adamsolomou/second-order-random-search.

`pysors` is my fork which adds scipy.minimize like interface

@inproceedings{
  lucchi2021randomsearch,
  title={On the Second-order Convergence Properties of Random Search Methods},
  author={Aurelien Lucchi and Antonio Orvieto and Adamos Solomou},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

LICENSE: Apache 2.0
"""

from .pysors_optimizer import *