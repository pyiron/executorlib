import contextlib
import importlib.util
import os
import queue
import sys
import time
from typing import Callable, Optional

from executorlib.standalone.cache import get_cache_files
from executorlib.standalone.command import get_command_path
from executorlib.standalone.interactive.communication import (
    SocketInterface,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.standalone.serialize import serialize_funct_h5


def execute_tasks(
    future_queue: queue.Queue,
    cores: int = 1,
    spawner: type[BaseSpawner] = MpiExecSpawner,
    hostname_localhost: Optional[bool] = None,
    init_function: Optional[Callable] = None,
    cache_directory: Optional[str] = None,
    cache_key: Optional[str] = None,
    queue_join_on_shutdown: bool = True,
    log_obj_size: bool = False,
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
    """
    interface = interface_bootup(
        command_lst=_get_backend_path(
            cores=cores,
        ),
        connections=spawner(cores=cores, **kwargs),
        hostname_localhost=hostname_localhost,
        log_obj_size=log_obj_size,
    )
    if init_function is not None:
        interface.send_dict(
            input_dict={"init": True, "fn": init_function, "args": (), "kwargs": {}}
        )
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            _task_done(future_queue=future_queue)
            if queue_join_on_shutdown:
                future_queue.join()
            break
        elif "fn" in task_dict and "future" in task_dict:
            if cache_directory is None:
                _execute_task_without_cache(
                    interface=interface, task_dict=task_dict, future_queue=future_queue
                )
            else:
                _execute_task_with_cache(
                    interface=interface,
                    task_dict=task_dict,
                    future_queue=future_queue,
                    cache_directory=cache_directory,
                    cache_key=cache_key,
                )


def _get_backend_path(
    cores: int,
) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        cores (int): Number of cores used to execute the task, if it is greater than one use interactive_parallel.py else interactive_serial.py

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1 and importlib.util.find_spec("mpi4py") is not None:
        command_lst += [get_command_path(executable="interactive_parallel.py")]
    elif cores > 1:
        raise ImportError(
            "mpi4py is required for parallel calculations. Please install mpi4py."
        )
    else:
        command_lst += [get_command_path(executable="interactive_serial.py")]
    return command_lst


def _execute_task_without_cache(
    interface: SocketInterface, task_dict: dict, future_queue: queue.Queue
):
    """
    Execute the task in the task_dict by communicating it via the interface.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_queue (Queue): Queue for receiving new tasks.
    """
    f = task_dict.pop("future")
    if not f.done() and f.set_running_or_notify_cancel():
        try:
            f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
        except Exception as thread_exception:
            interface.shutdown(wait=True)
            _task_done(future_queue=future_queue)
            f.set_exception(exception=thread_exception)
        else:
            _task_done(future_queue=future_queue)


def _execute_task_with_cache(
    interface: SocketInterface,
    task_dict: dict,
    future_queue: queue.Queue,
    cache_directory: str,
    cache_key: Optional[str] = None,
):
    """
    Execute the task in the task_dict by communicating it via the interface using the cache in the cache directory.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_queue (Queue): Queue for receiving new tasks.
        cache_directory (str): The directory to store cache files.
        cache_key (str, optional): By default the cache_key is generated based on the function hash, this can be
                                  overwritten by setting the cache_key.
    """
    from executorlib.task_scheduler.file.hdf import dump, get_output

    task_key, data_dict = serialize_funct_h5(
        fn=task_dict["fn"],
        fn_args=task_dict["args"],
        fn_kwargs=task_dict["kwargs"],
        resource_dict=task_dict.get("resource_dict", {}),
        cache_key=cache_key,
    )
    os.makedirs(cache_directory, exist_ok=True)
    file_name = os.path.abspath(os.path.join(cache_directory, task_key + "_o.h5"))
    if file_name not in get_cache_files(cache_directory=cache_directory):
        f = task_dict.pop("future")
        if f.set_running_or_notify_cancel():
            try:
                time_start = time.time()
                result = interface.send_and_receive_dict(input_dict=task_dict)
                data_dict["output"] = result
                data_dict["runtime"] = time.time() - time_start
                dump(file_name=file_name, data_dict=data_dict)
                f.set_result(result)
            except Exception as thread_exception:
                interface.shutdown(wait=True)
                _task_done(future_queue=future_queue)
                f.set_exception(exception=thread_exception)
                raise thread_exception
            else:
                _task_done(future_queue=future_queue)
    else:
        _, _, result = get_output(file_name=file_name)
        future = task_dict["future"]
        future.set_result(result)
        _task_done(future_queue=future_queue)


def _task_done(future_queue: queue.Queue):
    with contextlib.suppress(ValueError):
        future_queue.task_done()
