from ._version import get_versions
from pympipool.mpi.executor import PyMPIExecutor

try:  # The PyFluxExecutor requires flux-core to be installed.
    from pympipool.flux.executor import PyFluxExecutor
except ImportError:
    pass

try:  # The PySlurmExecutor requires the srun command to be available.
    from pympipool.slurm.executor import PySlurmExecutor
except ImportError:
    pass

__version__ = get_versions()["version"]
del get_versions
