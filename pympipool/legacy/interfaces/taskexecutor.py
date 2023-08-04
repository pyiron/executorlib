import os
import sys

from pympipool.shared.executorbase import (
    cloudpickle_register,
    execute_parallel_tasks_loop,
    ExecutorBase,
)
from pympipool.shared.thread import RaisingThread
from pympipool.legacy.shared.connections import interface_bootup


class Executor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

    Examples:
        ```
        >>> import numpy as np
        >>> from pympipool import Executor
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
        >>> with Executor(cores=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        cores,
        gpus_per_task=0,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_slurm_backend=False,
        init_function=None,
        cwd=None,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                "future_queue": self._future_queue,
                "cores": cores,
                "gpus_per_task": gpus_per_task,
                "oversubscribe": oversubscribe,
                "enable_flux_backend": enable_flux_backend,
                "enable_slurm_backend": enable_slurm_backend,
                "cwd": cwd,
                "queue_adapter": queue_adapter,
                "queue_adapter_kwargs": queue_adapter_kwargs,
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
    gpus_per_task=0,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    cwd=None,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
       oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
       enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
       enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
       cwd (str/None): current working directory where the parallel python task is executed
       queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
       queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter
    """
    command_lst = [
        sys.executable,
        os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "backend", "mpiexec.py")
        ),
    ]
    interface = interface_bootup(
        command_lst=command_lst,
        cwd=cwd,
        cores=cores,
        gpus_per_core=gpus_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_slurm_backend=enable_slurm_backend,
        queue_adapter=queue_adapter,
        queue_adapter_kwargs=queue_adapter_kwargs,
    )
    execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)
