import queue
from concurrent.futures import Future
from threading import Thread
from typing import Callable, Optional

from executorlib.standalone.inputcheck import (
    check_resource_dict,
    check_resource_dict_is_empty,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.standalone.queue import cancel_items_in_queue
from executorlib.task_scheduler.base import TaskSchedulerBase
from executorlib.task_scheduler.interactive.shared import execute_tasks


class BlockAllocationTaskScheduler(TaskSchedulerBase):
    """
    The executorlib.interactive.executor.InteractiveExecutor leverages the exeutorlib executor to distribute python
    tasks on a workstation or inside a queuing system allocation. In contrast to the mpi4py.futures.MPIPoolExecutor the
    executorlib.interactive.executor.InteractiveExecutor can be executed in a serial python process and does not require
    the python script to be executed with MPI. Consequently, it is primarily an abstraction of its functionality to
    improves the usability in particular when used in combination with Jupyter notebooks.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        executor_kwargs (dict): keyword arguments for the executor
        spawner (BaseSpawner): interface class to initiate python processes

    Examples:

        >>> import numpy as np
        >>> from executorlib.interactive.blockallocation import BlockAllocationTaskScheduler
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
        >>> with BlockAllocationTaskScheduler(max_workers=2, executor_kwargs={"init_function": init_k}) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_workers: int = 1,
        executor_kwargs: Optional[dict] = None,
        spawner: type[BaseSpawner] = MpiExecSpawner,
    ):
        if executor_kwargs is None:
            executor_kwargs = {}
        super().__init__(max_cores=executor_kwargs.get("max_cores"))
        executor_kwargs["future_queue"] = self._future_queue
        executor_kwargs["spawner"] = spawner
        executor_kwargs["queue_join_on_shutdown"] = False
        self._process_kwargs = executor_kwargs
        self._max_workers = max_workers
        self._set_process(
            process=[
                Thread(
                    target=execute_tasks,
                    kwargs=executor_kwargs | {"worker_id": worker_id},
                )
                for worker_id in range(self._max_workers)
            ],
        )

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, max_workers: int):
        if isinstance(self._future_queue, queue.Queue) and isinstance(
            self._process, list
        ):
            if self._max_workers > max_workers:
                for _ in range(self._max_workers - max_workers):
                    self._future_queue.queue.insert(0, {"shutdown": True, "wait": True})
                while len(self._process) > max_workers:
                    self._process = [
                        process for process in self._process if process.is_alive()
                    ]
            elif self._max_workers < max_workers:
                new_process_lst = [
                    Thread(
                        target=execute_tasks,
                        kwargs=self._process_kwargs,
                    )
                    for _ in range(max_workers - self._max_workers)
                ]
                for process_instance in new_process_lst:
                    process_instance.start()
                self._process += new_process_lst
            self._max_workers = max_workers

    def submit(  # type: ignore
        self, fn: Callable, *args, resource_dict: Optional[dict] = None, **kwargs
    ) -> Future:
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (Callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and
                                                              SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.

        Returns:
            Future: A Future representing the given call.
        """
        if resource_dict is None:
            resource_dict = {}
        check_resource_dict_is_empty(resource_dict=resource_dict)
        check_resource_dict(function=fn)
        f: Future = Future()
        if self._future_queue is not None:
            self._future_queue.put(
                {"fn": fn, "args": args, "kwargs": kwargs, "future": f}
            )
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                parallel_executors have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        if self._future_queue is not None:
            if cancel_futures:
                cancel_items_in_queue(que=self._future_queue)
            if isinstance(self._process, list):
                for _ in range(len(self._process)):
                    self._future_queue.put({"shutdown": True, "wait": wait})
                if wait:
                    for process in self._process:
                        process.join()
                    self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: list[Thread]):  # type: ignore
        """
        Set the process for the executor.

        Args:
            process (List[RaisingThread]): The process for the executor.
        """
        self._process = process
        for process_instance in self._process:
            process_instance.start()
