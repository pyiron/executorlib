import shutil
import subprocess


from pympipool.shared.executorbase import (
    cloudpickle_register,
    execute_parallel_tasks,
    ExecutorBase,
    executor_broker,
)
from pympipool.shared.interface import SrunInterface, SLURM_COMMAND
from pympipool.shared.thread import RaisingThread


if shutil.which(SLURM_COMMAND) is None:
    raise ImportError("SLURM command " + SLURM_COMMAND + " not found.")


class PySlurmExecutor(ExecutorBase):
    """
    The pympipool.slurm.PySlurmExecutor leverages the srun command to distribute python tasks within a SLURM queuing
    system allocation. In analogy to the pympipool.flux.PyFluxExecutor it provides the option to specify the number of
    threads per worker as well as the number of GPUs per worker in addition to specifying the number of cores per
    worker.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        sleep_interval (float): synchronization interval - default 0.1

    Examples:
        ```
        >>> import numpy as np
        >>> from pympipool.slurm import PySlurmExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> def init_k():
        >>>     return {"k": 3}
        >>>
        >>> with PySlurmExecutor(cores=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        max_workers=1,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                # Broker Arguments
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "sleep_interval": sleep_interval,
                "executor_class": PySlurmSingleTaskExecutor,
                # Executor Arguments
                "cores": cores_per_worker,
                "threads_per_core": threads_per_core,
                "gpus_per_task": int(gpus_per_worker / cores_per_worker),
                "oversubscribe": oversubscribe,
                "init_function": init_function,
                "cwd": cwd,
            },
        )
        self._process.start()


class PySlurmSingleTaskExecutor(ExecutorBase):
    """
    The pympipool.slurm.PySlurmSingleTaskExecutor is the internal worker for the pympipool.slurm.PySlurmExecutor.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed

    """

    def __init__(
        self,
        cores=1,
        threads_per_core=1,
        gpus_per_task=0,
        oversubscribe=False,
        init_function=None,
        cwd=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                # Executor Arguments
                "future_queue": self._future_queue,
                "cores": cores,
                "interface_class": SrunInterface,
                # Interface Arguments
                "threads_per_core": threads_per_core,
                "gpus_per_core": gpus_per_task,
                "cwd": cwd,
                "oversubscribe": oversubscribe,
            },
        )
        self._process.start()
        self._set_init_function(init_function=init_function)
        cloudpickle_register(ind=3)
