"""
Up-scale python functions for high performance computing (HPC) with executorlib. The executorlib module provides five
different executor classes, namely:
* SingleNodeExecutor - for testing executorlib on your local workstation, before up-scaling to HPC.
* SlurmClusterExecutor - for SLURM clusters, submitting Python functions as SLURM jobs.
* FluxClusterExecutor - for flux-framework clusters, submitting Python functions as flux jobs.
* SlurmJobExecutor - for distributing Python functions within a given SLRUM job.
* FluxJobExecutor - for distributing Python functions within a given flux job or SLRUM job.

In addition, the executorlib includes a BaseExecutor class to validate a given executor object is based on executorlib.
Finally, the get_cache_data() function allows users to cache the content of their current cache directory in one
pandas.DataFrame.
"""

from executorlib.executor.base import BaseExecutor
from executorlib.executor.flux import (
    FluxClusterExecutor,
    FluxJobExecutor,
)
from executorlib.executor.single import SingleNodeExecutor
from executorlib.executor.slurm import (
    SlurmClusterExecutor,
    SlurmJobExecutor,
)
from executorlib.standalone.cache import get_cache_data

from . import _version

__all__: list[str] = [
    "get_cache_data",
    "BaseExecutor",
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]

__version__ = _version.__version__
