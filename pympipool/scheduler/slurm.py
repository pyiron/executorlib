from typing import Optional
from pympipool.shared.executorbase import (
    execute_parallel_tasks,
    execute_separate_tasks,
    ExecutorBroker,
    ExecutorSteps,
)
from pympipool.shared.interface import SrunInterface
from pympipool.shared.thread import RaisingThread


class PySlurmExecutor(ExecutorBroker):
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
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        command_line_argument_lst (list): Additional command line arguments for the srun call

    Examples:

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
        >>> with PySlurmExecutor(max_workers=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
    """

    def __init__(
        self,
        max_workers: int = 1,
        cores_per_worker: int = 1,
        threads_per_core: int = 1,
        gpus_per_worker: int = 0,
        oversubscribe: bool = False,
        init_function: Optional[callable] = None,
        cwd: Optional[str] = None,
        hostname_localhost: bool = False,
        command_line_argument_lst: list[str] = [],
    ):
        super().__init__()
        self._set_process(
            process=[
                RaisingThread(
                    target=execute_parallel_tasks,
                    kwargs={
                        # Executor Arguments
                        "future_queue": self._future_queue,
                        "cores": cores_per_worker,
                        "interface_class": SrunInterface,
                        "hostname_localhost": hostname_localhost,
                        "init_function": init_function,
                        # Interface Arguments
                        "threads_per_core": threads_per_core,
                        "gpus_per_core": int(gpus_per_worker / cores_per_worker),
                        "cwd": cwd,
                        "oversubscribe": oversubscribe,
                        "command_line_argument_lst": command_line_argument_lst,
                    },
                )
                for _ in range(max_workers)
            ],
        )


class PySlurmStepExecutor(ExecutorSteps):
    """
    The pympipool.slurm.PySlurmStepExecutor leverages the srun command to distribute python tasks within a SLURM queuing
    system allocation. In analogy to the pympipool.flux.PyFluxExecutor it provides the option to specify the number of
    threads per worker as well as the number of GPUs per worker in addition to specifying the number of cores per
    worker.

    Args:
        max_cores (int): defines the number cores which can be used in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        cwd (str/None): current working directory where the parallel python task is executed
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        command_line_argument_lst (list): Additional command line arguments for the srun call

    Examples:

        >>> import numpy as np
        >>> from pympipool.slurm import PySlurmStepExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> with PySlurmStepExecutor(max_cores=2) as p:
        >>>     fs = p.submit(calc, 2, j=4, k=3, resource_dict={"cores": 2})
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
    """

    def __init__(
        self,
        max_cores: int = 1,
        cores_per_worker: int = 1,
        threads_per_core: int = 1,
        gpus_per_worker: int = 0,
        oversubscribe: bool = False,
        cwd: Optional[str] = None,
        hostname_localhost: bool = False,
        command_line_argument_lst: list[str] = [],
    ):
        super().__init__()
        self._set_process(
            RaisingThread(
                target=execute_separate_tasks,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "cores": cores_per_worker,
                    "interface_class": SrunInterface,
                    "max_cores": max_cores,
                    "hostname_localhost": hostname_localhost,
                    # Interface Arguments
                    "threads_per_core": threads_per_core,
                    "gpus_per_core": int(gpus_per_worker / cores_per_worker),
                    "cwd": cwd,
                    "oversubscribe": oversubscribe,
                    "command_line_argument_lst": command_line_argument_lst,
                },
            )
        )
