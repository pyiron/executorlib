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

from concurrent.futures import Future
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
from executorlib.standalone.select import get_item_from_future, split_future


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


def get_future_from_cache(
    cache_directory: str,
    cache_key: str,
) -> Future:
    """
    Reload future from HDF5 file in cache directory with the given cache key. The function checks if the output file
    exists, if not it checks for the input file. If neither of them exist, it raises a FileNotFoundError. If the output
    file exists, it loads the output and sets it as the result of the future. If only the input file exists, it checks
    if the execution is finished and if there was an error. If there was no error, it sets the output as the result of
    the future, otherwise it raises the error.

    Args:
        cache_directory (str): The directory to store cache files.
        cache_key (str): The key of the cache file to be reloaded.

    Returns:
        Future: Future object containing the result of the execution of the python function.
    """
    from executorlib.standalone.hdf import get_future_from_cache

    return get_future_from_cache(
        cache_directory=cache_directory,
        cache_key=cache_key,
    )


def terminate_tasks_in_cache(
    cache_directory: str,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Delete all jobs stored in the cache directory from the queuing system

    Args:
        cache_directory (str): The directory to store cache files.
        pysqa_config_directory (str, optional): path to the pysqa config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
    """
    from executorlib.task_scheduler.file.spawner_pysqa import terminate_tasks_in_cache

    return terminate_tasks_in_cache(
        cache_directory=cache_directory,
        pysqa_config_directory=pysqa_config_directory,
        backend=backend,
    )


def terminate_task_in_cache(
    cache_directory: str,
    cache_key: str,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Delete a specific job stored in the cache directory from the queuing system

    Args:
        cache_directory (str): The directory to store cache files.
        cache_key (str): The key of the cache file to be deleted.
        pysqa_config_directory (str, optional): path to the pysqa config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
    """
    from executorlib.task_scheduler.file.spawner_pysqa import terminate_task_in_cache

    return terminate_task_in_cache(
        cache_directory=cache_directory,
        cache_key=cache_key,
        pysqa_config_directory=pysqa_config_directory,
        backend=backend,
    )


__all__: list[str] = [
    "get_cache_data",
    "get_future_from_cache",
    "get_item_from_future",
    "split_future",
    "terminate_tasks_in_cache",
    "BaseExecutor",
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]

__version__ = executorlib._version.__version__
