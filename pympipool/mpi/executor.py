from pympipool.shared.executorbase import (
    cloudpickle_register,
    execute_parallel_tasks,
    ExecutorBase,
    executor_broker,
)
from pympipool.shared.interface import MpiExecInterface
from pympipool.shared.thread import RaisingThread


class PyMPIExecutor(ExecutorBase):
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
        max_workers=1,
        cores_per_worker=1,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        hostname_localhost=False,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                # Broker Arguments
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "executor_class": PyMPISingleTaskExecutor,
                "hostname_localhost": hostname_localhost,
                # Executor Arguments
                "cores": cores_per_worker,
                "oversubscribe": oversubscribe,
                "init_function": init_function,
                "cwd": cwd,
            },
        )
        self._process.start()


class PyMPISingleTaskExecutor(ExecutorBase):
    """
    The pympipool.mpi.PyMPISingleTaskExecutor is the internal worker for the pympipool.mpi.PyMPIExecutor.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
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

    """

    def __init__(
        self,
        cores=1,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        hostname_localhost=False,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                # Executor Arguments
                "future_queue": self._future_queue,
                "cores": cores,
                "interface_class": MpiExecInterface,
                # Interface Arguments
                "cwd": cwd,
                "oversubscribe": oversubscribe,
                "hostname_localhost": hostname_localhost,
            },
        )
        self._process.start()
        self._set_init_function(init_function=init_function)
        cloudpickle_register(ind=3)
