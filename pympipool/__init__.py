from ._version import get_versions
from pympipool.mpi.executor import PyMPIExecutor
from pympipool.slurm.executor import PySlurmExecutor

try:  # The PyFluxExecutor requires flux-core to be installed.
    from pympipool.flux.executor import PyFluxExecutor
except ImportError:
    pass

__version__ = get_versions()["version"]
del get_versions
