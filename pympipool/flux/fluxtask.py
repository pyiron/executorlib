import os

import flux.job

from pympipool.shared.executorbase import (
    cloudpickle_register,
    ExecutorBase,
    execute_parallel_tasks_loop,
    get_backend_path,
)
from pympipool.shared.interface import BaseInterface
from pympipool.shared.communication import interface_bootup
from pympipool.shared.thread import RaisingThread


class PyFluxSingleTaskExecutor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed

    Examples:
        ```
        >>> import numpy as np
        >>> from pympipool.flux.fluxtask import PyFluxSingleTaskExecutor
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
        >>> with PyFluxSingleTaskExecutor(cores=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        init_function=None,
        **kwargs,
    ):
        super().__init__()
        executor_kwargs = {
            "future_queue": self._future_queue,
        }
        executor_kwargs.update(kwargs)
        self._process = RaisingThread(
            target=_flux_execute_parallel_tasks,
            kwargs=executor_kwargs,
        )
        self._process.start()
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )
        cloudpickle_register(ind=3)


class FluxPythonInterface(BaseInterface):
    def __init__(self, executor=None, **kwargs):
        super().__init__(**kwargs)
        self._executor = executor
        self._future = None

    def bootup(self, command_lst):
        if self._oversubscribe:
            raise ValueError(
                "Oversubscribing is currently not supported for the Flux adapter."
            )
        if self._executor is None:
            self._executor = flux.job.FluxExecutor()
        jobspec = flux.job.JobspecV1.from_command(
            command=command_lst,
            num_tasks=self._cores,
            cores_per_task=self._threads_per_core,
            gpus_per_task=self._gpus_per_core,
            num_nodes=None,
            exclusive=False,
        )
        jobspec.environment = dict(os.environ)
        if self._cwd is not None:
            jobspec.cwd = self._cwd
        self._future = self._executor.submit(jobspec)

    def shutdown(self, wait=True):
        if self.poll():
            self._future.cancel()
        # The flux future objects are not instantly updated,
        # still showing running after cancel was called,
        # so we wait until the execution is completed.
        self._future.result()

    def poll(self):
        return self._future is not None and not self._future.done()


def _flux_execute_parallel_tasks(
    future_queue,
    cores,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       threads_per_core (int): number of OpenMP threads to be used for each function call
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
       cwd (str/None): current working directory where the parallel python task is executed
       executor (flux.job.FluxExecutor/None): flux executor to submit tasks to - optional
    """
    execute_parallel_tasks_loop(
        interface=interface_bootup(
            command_lst=get_backend_path(cores=cores),
            connections=FluxPythonInterface(cores=cores, **kwargs),
        ),
        future_queue=future_queue,
    )
