import os
from socket import gethostname
import sys

from pympipool.shared.executorbase import (
    cloudpickle_register,
    execute_parallel_tasks_loop,
    ExecutorBase,
)
from pympipool.shared.thread import RaisingThread
from pympipool.shared.communication import SocketInterface
from pympipool.shared.interface import MpiExecInterface


class MPISingleTaskExecutor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed

    Examples:
        ```
        >>> import numpy as np
        >>> from pympipool import MPISingleTaskExecutor
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
        >>> with MPISingleTaskExecutor(cores=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        cores,
        oversubscribe=False,
        init_function=None,
        cwd=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                "future_queue": self._future_queue,
                "cores": cores,
                "cwd": cwd,
                "oversubscribe": oversubscribe,
            },
        )
        self._process.start()
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )
        cloudpickle_register(ind=3)


def execute_parallel_tasks(
    future_queue,
    cores,
    cwd=None,
    oversubscribe=False,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       cwd (str/None): current working directory where the parallel python task is executed
       oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
    """
    command_lst = [
        sys.executable,
        os.path.abspath(os.path.join(__file__, "..", "..", "backend", "mpiexec.py")),
    ]
    interface = interface_bootup(
        command_lst=command_lst,
        cores=cores,
        cwd=cwd,
        oversubscribe=oversubscribe,
    )
    execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def interface_bootup(
    command_lst,
    cores=1,
    cwd=None,
    oversubscribe=False,
):
    command_lst += [
        "--host",
        gethostname(),
    ]
    connections = MpiExecInterface(
        cwd=cwd,
        cores=cores,
        threads_per_core=1,
        gpus_per_core=0,
        oversubscribe=oversubscribe,
    )
    interface = SocketInterface(interface=connections)
    command_lst += [
        "--zmqport",
        str(interface.bind_to_random_port()),
    ]
    interface.bootup(command_lst=command_lst)
    return interface
