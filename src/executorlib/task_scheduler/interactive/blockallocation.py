import queue
import random
from concurrent.futures import Future
from threading import Thread
from typing import Callable, Optional

from executorlib.standalone.command import get_interactive_execute_command
from executorlib.standalone.inputcheck import (
    check_resource_dict,
    check_resource_dict_is_empty,
)
from executorlib.standalone.interactive.communication import (
    ExecutorlibSocketError,
    SocketInterface,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.standalone.queue import cancel_items_in_queue
from executorlib.task_scheduler.base import TaskSchedulerBase
from executorlib.task_scheduler.interactive.shared import (
    execute_task_dict,
    reset_task_dict,
    task_done,
)

_interrupt_bootup_dict: dict = {}


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
        >>> from executorlib.task_scheduler.interactive.blockallocation import BlockAllocationTaskScheduler
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
        self_id = random.getrandbits(128)
        self._self_id = self_id
        _interrupt_bootup_dict[self._self_id] = False
        self._set_process(
            process=[
                Thread(
                    target=_execute_multiple_tasks,
                    kwargs=executor_kwargs
                    | {
                        "worker_id": worker_id,
                        "stop_function": lambda: _interrupt_bootup_dict[self_id],
                    },
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
                        target=_execute_multiple_tasks,
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
                                  - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI
                                                                  and SLURM only) - default False
                                  - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM
                                                           only)
                                  - error_log_file (str): Name of the error log file to use for storing exceptions
                                                          raised by the Python functions submitted to the Executor.

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
            wait (bool): If True then shutdown will not return until all running futures have finished executing and
                         the resources used by the parallel_executors have been reclaimed.
            cancel_futures (bool): If True then shutdown will cancel all pending futures. Futures that are completed or
                                   running will not be cancelled.
        """
        if self._future_queue is not None:
            if cancel_futures:
                cancel_items_in_queue(que=self._future_queue)
            if isinstance(self._process, list):
                _interrupt_bootup_dict[self._self_id] = True
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


def _execute_multiple_tasks(
    future_queue: queue.Queue,
    cores: int = 1,
    spawner: type[BaseSpawner] = MpiExecSpawner,
    hostname_localhost: Optional[bool] = None,
    init_function: Optional[Callable] = None,
    cache_directory: Optional[str] = None,
    cache_key: Optional[str] = None,
    queue_join_on_shutdown: bool = True,
    log_obj_size: bool = False,
    error_log_file: Optional[str] = None,
    worker_id: Optional[int] = None,
    stop_function: Optional[Callable] = None,
    restart_limit: int = 0,
    **kwargs,
) -> None:
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
        future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
        cores (int): defines the total number of MPI ranks to use
        spawner (BaseSpawner): Spawner to start process on selected compute resources
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        init_function (Callable): optional function to preset arguments for functions which are submitted later
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        cache_key (str, optional): By default the cache_key is generated based on the function hash, this can be
                                   overwritten by setting the cache_key.
        queue_join_on_shutdown (bool): Join communication queue when thread is closed. Defaults to True.
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
        error_log_file (str): Name of the error log file to use for storing exceptions raised by the Python functions
                              submitted to the Executor.
        worker_id (int): Communicate the worker which ID was assigned to it for future reference and resource
                         distribution.
        stop_function (Callable): Function to stop the interface.
        restart_limit (int): The maximum number of restarting worker processes.
    """
    interface = interface_bootup(
        command_lst=get_interactive_execute_command(
            cores=cores,
        ),
        connections=spawner(cores=cores, **kwargs),
        hostname_localhost=hostname_localhost,
        log_obj_size=log_obj_size,
        worker_id=worker_id,
        stop_function=stop_function,
    )
    interface_initialization_exception = _set_init_function(
        interface=interface,
        init_function=init_function,
    )
    restart_counter = 0
    while True:
        if not interface.status and restart_counter > restart_limit:
            interface.status = True  # no more restarts
            interface_initialization_exception = ExecutorlibSocketError(
                "SocketInterface crashed during execution."
            )
        elif not interface.status:
            interface.bootup()
            interface_initialization_exception = _set_init_function(
                interface=interface,
                init_function=init_function,
            )
            restart_counter += 1
        else:  # interface.status == True
            task_dict = future_queue.get()
            if "shutdown" in task_dict and task_dict["shutdown"]:
                if interface.status:
                    interface.shutdown(wait=task_dict["wait"])
                task_done(future_queue=future_queue)
                if queue_join_on_shutdown:
                    future_queue.join()
                break
            elif "fn" in task_dict and "future" in task_dict:
                f = task_dict.pop("future")
                if interface_initialization_exception is not None:
                    f.set_exception(exception=interface_initialization_exception)
                else:
                    # The interface failed during the execution
                    interface.status = execute_task_dict(
                        task_dict=task_dict,
                        future_obj=f,
                        interface=interface,
                        cache_directory=cache_directory,
                        cache_key=cache_key,
                        error_log_file=error_log_file,
                    )
                    if not interface.status:
                        reset_task_dict(
                            future_obj=f, future_queue=future_queue, task_dict=task_dict
                        )
                task_done(future_queue=future_queue)


def _set_init_function(
    interface: SocketInterface,
    init_function: Optional[Callable] = None,
) -> Optional[Exception]:
    interface_initialization_exception = None
    if init_function is not None and interface.status:
        try:
            _ = interface.send_and_receive_dict(
                input_dict={"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )
        except Exception as init_exception:
            interface_initialization_exception = init_exception
    return interface_initialization_exception
