from concurrent.futures import Executor as FutureExecutor, Future
from queue import Queue
from threading import Thread

from pympipool.share.serial import (
    execute_parallel_tasks,
    execute_serial_tasks,
    cloudpickle_register,
    cancel_items_in_queue,
)


class ExecutorBase(FutureExecutor):
    def __init__(self):
        self._future_queue = Queue()
        self._process = None
        cloudpickle_register(ind=3)

    def submit(self, fn, *args, **kwargs):
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        f = Future()
        self._future_queue.put({"fn": fn, "args": args, "kwargs": kwargs, "future": f})
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        if cancel_futures:
            cancel_items_in_queue(que=self._future_queue)
        self._future_queue.put({"shutdown": True, "wait": wait})
        self._process.join()

    def __len__(self):
        return self._future_queue.qsize()


class Executor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed

    Simple example:
        ```
        import numpy as np
        from pympipool import Executor

        def calc(i, j, k):
            from mpi4py import MPI
            size = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
            return np.array([i, j, k]), size, rank

        def init_k():
            return {"k": 3}

        with Executor(cores=2, init_function=init_k) as p:
            fs = p.submit(calc, 2, j=4)
            print(fs.result())

        >>> [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        cores,
        oversubscribe=False,
        enable_flux_backend=False,
        init_function=None,
        cwd=None,
    ):
        super().__init__()
        self._process = Thread(
            target=execute_parallel_tasks,
            args=(self._future_queue, cores, oversubscribe, enable_flux_backend, cwd),
        )
        self._process.start()
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )


class PoolExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers=1,
        oversubscribe=False,
        enable_flux_backend=False,
        cwd=None,
    ):
        super().__init__()
        self._process = Thread(
            target=execute_serial_tasks,
            args=(
                self._future_queue,
                max_workers,
                oversubscribe,
                enable_flux_backend,
                cwd,
            ),
        )
        self._process.start()
