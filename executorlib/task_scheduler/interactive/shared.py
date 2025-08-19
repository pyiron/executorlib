import contextlib
import os
import queue
import time
from typing import Callable, Optional

from executorlib.standalone.command import get_interactive_execute_command
from executorlib.standalone.interactive.communication import (
    SocketInterface,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.standalone.serialize import serialize_funct


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
    error_log_file: Optional[str] = None,
    worker_id: Optional[int] = None,
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
    """
    interface = interface_bootup(
        command_lst=get_interactive_execute_command(
            cores=cores,
        ),
        connections=spawner(cores=cores, **kwargs),
        hostname_localhost=hostname_localhost,
        log_obj_size=log_obj_size,
        worker_id=worker_id,
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
            if error_log_file is not None:
                task_dict["error_log_file"] = error_log_file
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
    from executorlib.standalone.hdf import dump, get_cache_files, get_output

    task_key, data_dict = serialize_funct(
        fn=task_dict["fn"],
        fn_args=task_dict["args"],
        fn_kwargs=task_dict["kwargs"],
        resource_dict=task_dict.get("resource_dict", {}),
        cache_key=cache_key,
    )
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
