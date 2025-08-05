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

from typing import Optional

import executorlib._version
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


def get_cache_data(cache_directory: str) -> list[dict]:
    """
    Collect all HDF5 files in the cache directory

    Args:
        cache_directory (str): The directory to store cache files.

    Returns:
        list[dict]: List of dictionaries each representing on of the HDF5 files in the cache directory.
    """
    from executorlib.standalone.hdf import get_cache_data

    return get_cache_data(cache_directory=cache_directory)


def terminate_tasks_in_cache(
    cache_directory: str,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Delete all jobs stored in the cache directory from the queuing system

    Args:
        cache_directory (str): The directory to store cache files.
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
    """
    from executorlib.task_scheduler.file.queue_spawner import terminate_tasks_in_cache

    return terminate_tasks_in_cache(
        cache_directory=cache_directory,
        config_directory=config_directory,
        backend=backend,
    )


__all__: list[str] = [
    "get_cache_data",
    "terminate_tasks_in_cache",
    "BaseExecutor",
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]

__version__ = executorlib._version.__version__
