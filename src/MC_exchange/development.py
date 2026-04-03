import numpy as np
import warnings
import random

from itertools import combinations

import warnings

from mpi4py import MPI
import sys
from .calculations import calculate_distance_pbc, calculate_raw_fene_potential, calculate_lj_potential

from .error_handling import mpi_abort_on_exception

# Assign the hook at the very beginning of your script
sys.excepthook = mpi_abort_on_exception

