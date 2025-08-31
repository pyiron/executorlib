import contextlib
import os
import queue
import time
from typing import Optional
from concurrent.futures import Future

from executorlib.standalone.interactive.communication import SocketInterface
from executorlib.standalone.serialize import serialize_funct


def execute_task_dict(
    task_dict: dict,
    future_obj: Future,
    interface: SocketInterface,
    cache_directory: Optional[str] = None,
    cache_key: Optional[str] = None,
    error_log_file: Optional[str] = None,
):
    """
    Execute the task in the task_dict by communicating it via the interface.

    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_obj (Future): A Future representing the given call.
        interface (SocketInterface): socket interface for zmq communication
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        cache_key (str, optional): By default the cache_key is generated based on the function hash, this can be
                                  overwritten by setting the cache_key.
        error_log_file (str): Name of the error log file to use for storing exceptions raised by the Python functions
                              submitted to the Executor.
    """
    if error_log_file is not None:
        task_dict["error_log_file"] = error_log_file
    if cache_directory is None:
        _execute_task_without_cache(interface=interface, task_dict=task_dict, future_obj=future_obj)
    else:
        _execute_task_with_cache(
            interface=interface,
            task_dict=task_dict,
            cache_directory=cache_directory,
            cache_key=cache_key,
            future_obj=future_obj,
        )


def task_done(future_queue: queue.Queue):
    """
    Mark the current task as done in the current queue.

    Args:
        future_queue (queue): Queue of task dictionaries waiting for execution.
    """
    with contextlib.suppress(ValueError):
        future_queue.task_done()


def _execute_task_without_cache(interface: SocketInterface, task_dict: dict, future_obj: Future):
    """
    Execute the task in the task_dict by communicating it via the interface.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_obj (Future): A Future representing the given call.
    """
    if not future_obj.done() and future_obj.set_running_or_notify_cancel():
        try:
            future_obj.set_result(interface.send_and_receive_dict(input_dict=task_dict))
        except Exception as thread_exception:
            interface.shutdown(wait=True)
            future_obj.set_exception(exception=thread_exception)


def _execute_task_with_cache(
    interface: SocketInterface,
    task_dict: dict,
    future_obj: Future,
    cache_directory: str,
    cache_key: Optional[str] = None,
):
    """
    Execute the task in the task_dict by communicating it via the interface using the cache in the cache directory.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": Callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_obj (Future): A Future representing the given call.
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
        if future_obj.set_running_or_notify_cancel():
            try:
                time_start = time.time()
                result = interface.send_and_receive_dict(input_dict=task_dict)
                data_dict["output"] = result
                data_dict["runtime"] = time.time() - time_start
                dump(file_name=file_name, data_dict=data_dict)
                future_obj.set_result(result)
            except Exception as thread_exception:
                interface.shutdown(wait=True)
                future_obj.set_exception(exception=thread_exception)
    else:
        _, _, result = get_output(file_name=file_name)
        future_obj.set_result(result)
