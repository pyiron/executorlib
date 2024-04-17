from typing import Optional

from pympipool.shared.executorbase import (
    execute_parallel_tasks,
    execute_separate_tasks,
    ExecutorBroker,
    ExecutorSteps,
)
from pympipool.shared.interface import MpiExecInterface
from pympipool.shared.thread import RaisingThread


class PyMPIExecutor(ExecutorBroker):
    """
    The pympipool.mpi.PyMPIExecutor leverages the message passing interface MPI to distribute python tasks within an
    MPI allocation. In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.mpi.PyMPIExecutor can be executed
    in a serial python process and does not require the python script to be executed with MPI. Consequently, it is
    primarily an abstraction of its functionality to improve the usability in particular when used in combination with \
    Jupyter notebooks.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
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

    Examples:

        >>> import numpy as np
        >>> from pympipool.mpi import PyMPIExecutor
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
        >>> with PyMPIExecutor(max_workers=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_workers: int = 1,
        cores_per_worker: int = 1,
        oversubscribe: bool = False,
        init_function: Optional[callable] = None,
        cwd: Optional[str] = None,
        hostname_localhost: bool = False,
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
                        "interface_class": MpiExecInterface,
                        "hostname_localhost": hostname_localhost,
                        "init_function": init_function,
                        # Interface Arguments
                        "cwd": cwd,
                        "oversubscribe": oversubscribe,
                    },
                )
                for _ in range(max_workers)
            ],
        )


class PyMPIStepExecutor(ExecutorSteps):
    """
    The pympipool.mpi.PyMPIStepExecutor leverages the message passing interface MPI to distribute python tasks within an
    MPI allocation. In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.mpi.PyMPIStepExecutor can be executed
    in a serial python process and does not require the python script to be executed with MPI. Consequently, it is
    primarily an abstraction of its functionality to improve the usability in particular when used in combination with \
    Jupyter notebooks.

    Args:
        max_cores (int): defines the number cores which can be used in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        cwd (str/None): current working directory where the parallel python task is executed
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true

    Examples:

        >>> import numpy as np
        >>> from pympipool.mpi import PyMPIStepExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> with PyMPIStepExecutor(max_cores=2) as p:
        >>>     fs = p.submit(calc, 2, j=4, k=3, resource_dict={"cores": 2})
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_cores: int = 1,
        cores_per_worker: int = 1,
        oversubscribe: bool = False,
        cwd: Optional[str] = None,
        hostname_localhost: bool = False,
    ):
        super().__init__()
        self._set_process(
            RaisingThread(
                target=execute_separate_tasks,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "cores": cores_per_worker,
                    "interface_class": MpiExecInterface,
                    "max_cores": max_cores,
                    "hostname_localhost": hostname_localhost,
                    # Interface Arguments
                    "cwd": cwd,
                    "oversubscribe": oversubscribe,
                },
            )
        )
