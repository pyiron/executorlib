import queue
from concurrent.futures import Future
from threading import Thread
from typing import Optional

from executorlib.standalone.command import get_interactive_execute_command
from executorlib.standalone.interactive.communication import (
    ExecutorlibSocketError,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.task_scheduler.base import TaskSchedulerBase
from executorlib.task_scheduler.interactive.shared import execute_task_dict


class OneProcessTaskScheduler(TaskSchedulerBase):
    """
    The executorlib.interactive.executor.InteractiveStepExecutor leverages the executorlib executor to distribute python
    tasks. In contrast to the mpi4py.futures.MPIPoolExecutor the executorlib.interactive.executor.InteractiveStepExecutor
    can be executed in a serial python process and does not require the python script to be executed with MPI.
    Consequently, it is primarily an abstraction of its functionality to improve the usability in particular when used
    in combination with Jupyter notebooks.

    Args:
        max_cores (int): defines the number workers which can execute functions in parallel
        executor_kwargs (dict): keyword arguments for the executor
        spawner (BaseSpawner): interface class to initiate python processes

    Examples:

        >>> import numpy as np
        >>> from executorlib.task_scheduler.interactive.onetoone import OneProcessTaskScheduler
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> with OneProcessTaskScheduler(max_cores=2) as p:
        >>>     fs = p.submit(calc, 2, j=4, k=3, resource_dict={"cores": 2})
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_cores: Optional[int] = None,
        max_workers: Optional[int] = None,
        executor_kwargs: Optional[dict] = None,
        spawner: type[BaseSpawner] = MpiExecSpawner,
    ):
        if executor_kwargs is None:
            executor_kwargs = {}
        super().__init__(max_cores=executor_kwargs.get("max_cores"))
        executor_kwargs.update(
            {
                "future_queue": self._future_queue,
                "spawner": spawner,
                "max_cores": max_cores,
                "max_workers": max_workers,
            }
        )
        self._process_kwargs = executor_kwargs
        self._set_process(
            Thread(
                target=_execute_single_task,
                kwargs=executor_kwargs,
            )
        )


def _execute_single_task(
    future_queue: queue.Queue,
    spawner: type[BaseSpawner] = MpiExecSpawner,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
    hostname_localhost: Optional[bool] = None,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
        future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
        spawner (BaseSpawner): Interface to start process on selected compute resources
        max_cores (int): defines the number cores which can be used in parallel
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
    """
    active_task_dict: dict = {}
    process_lst: list = []
    if "cores" not in kwargs:
        kwargs["cores"] = 1
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict and task_dict["shutdown"]:
            if task_dict["wait"]:
                _ = [process.join() for process in process_lst]
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict and "future" in task_dict:
            process, active_task_dict = _wrap_execute_task_in_separate_process(
                task_dict=task_dict,
                active_task_dict=active_task_dict,
                spawner=spawner,
                executor_kwargs=kwargs,
                max_cores=max_cores,
                max_workers=max_workers,
                hostname_localhost=hostname_localhost,
            )
            process_lst.append(process)
            future_queue.task_done()


def _wait_for_free_slots(
    active_task_dict: dict,
    cores_requested: int,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Wait for available computing resources to become available.

    Args:
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        cores_requested (int): Number of cores required for executing the next task
        max_cores (int): Maximum number cores which can be used
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.

    Returns:
        dict: Dictionary containing the future objects and the number of cores they require
    """
    if max_cores is not None:
        while sum(active_task_dict.values()) + cores_requested > max_cores:
            active_task_dict = {
                k: v for k, v in active_task_dict.items() if not k.done()
            }
    elif max_workers is not None and max_cores is None:
        while len(active_task_dict.values()) + 1 > max_workers:
            active_task_dict = {
                k: v for k, v in active_task_dict.items() if not k.done()
            }
    return active_task_dict


def _wrap_execute_task_in_separate_process(
    task_dict: dict,
    active_task_dict: dict,
    spawner: type[BaseSpawner],
    executor_kwargs: dict,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
    hostname_localhost: Optional[bool] = None,
):
    """
    Submit function to be executed in separate Python process
    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        spawner (BaseSpawner): Interface to start process on selected compute resources
        executor_kwargs (dict): keyword parameters used to initialize the Executor
        max_cores (int): defines the number cores which can be used in parallel
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
    Returns:
        RaisingThread, dict: thread for communicating with the python process which is executing the function and
                             dictionary containing the future objects and the number of cores they require
    """
    resource_dict = task_dict.pop("resource_dict").copy()
    f = task_dict.pop("future")
    if "cores" not in resource_dict or (
        resource_dict["cores"] == 1 and executor_kwargs["cores"] >= 1
    ):
        resource_dict["cores"] = executor_kwargs["cores"]
    slots_required = resource_dict["cores"] * resource_dict.get("threads_per_core", 1)
    active_task_dict = _wait_for_free_slots(
        active_task_dict=active_task_dict,
        cores_requested=slots_required,
        max_cores=max_cores,
        max_workers=max_workers,
    )
    active_task_dict[f] = slots_required
    task_kwargs = executor_kwargs.copy()
    task_kwargs.update(resource_dict)
    task_kwargs.update(
        {
            "task_dict": task_dict,
            "spawner": spawner,
            "hostname_localhost": hostname_localhost,
            "future_obj": f,
        }
    )
    process = Thread(
        target=_execute_task_in_thread,
        kwargs=task_kwargs,
    )
    process.start()
    return process, active_task_dict


def _execute_task_in_thread(
    task_dict: dict,
    future_obj: Future,
    cores: int = 1,
    spawner: type[BaseSpawner] = MpiExecSpawner,
    hostname_localhost: Optional[bool] = None,
    cache_directory: Optional[str] = None,
    cache_key: Optional[str] = None,
    log_obj_size: bool = False,
    error_log_file: Optional[str] = None,
    worker_id: Optional[int] = None,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_obj (Future): A Future representing the given call.
        cores (int): defines the total number of MPI ranks to use
        spawner (BaseSpawner): Spawner to start process on selected compute resources
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        cache_key (str, optional): By default the cache_key is generated based on the function hash, this can be
                                   overwritten by setting the cache_key.
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
        error_log_file (str): Name of the error log file to use for storing exceptions raised by the Python functions
                              submitted to the Executor.
        worker_id (int): Communicate the worker which ID was assigned to it for future reference and resource
                         distribution.
    """
    if not execute_task_dict(
        task_dict=task_dict,
        future_obj=future_obj,
        interface=interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=cores,
            ),
            connections=spawner(cores=cores, **kwargs),
            hostname_localhost=hostname_localhost,
            log_obj_size=log_obj_size,
            worker_id=worker_id,
        ),
        cache_directory=cache_directory,
        cache_key=cache_key,
        error_log_file=error_log_file,
    ):
        future_obj.set_exception(
            ExecutorlibSocketError("SocketInterface crashed during execution.")
        )
