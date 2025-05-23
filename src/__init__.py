"""
Federated Learning Prototype with SMC and Differential Privacy

A hybrid approach to privacy-preserving federated learning that integrates
Secure Multiparty Computation (SMC) and Differential Privacy (DP).
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from . import aggregator, communication, data_party, dp, models, smc, utils
from .utils import data_loader
