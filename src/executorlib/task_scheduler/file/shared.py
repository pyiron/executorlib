import contextlib
import os
import queue
from concurrent.futures import Future
from time import sleep
from typing import Any, Callable, Optional

from executorlib.standalone.command import get_cache_execute_command
from executorlib.standalone.hdf import get_cache_files, get_output, get_queue_id
from executorlib.standalone.serialize import serialize_funct
from executorlib.task_scheduler.file.spawner_subprocess import subprocess_terminate


class FutureItem:
    def __init__(self, file_name: str, selector: Optional[int | str] = None):
        """
        Initialize a FutureItem object.

        Args:
            file_name (str): The name of the file.
            selector (int | str, optional): The selector to select a specific part of the result. Defaults to None.

        """
        self._file_name = file_name
        self._selector = selector

    def result(self) -> Any:
        """
        Get the result of the future item.

        Returns:
            str: The result of the future item.

        """
        exec_flag, no_error_flag, result = get_output(file_name=self._file_name)
        if exec_flag and no_error_flag:
            if self._selector is not None:
                return result[self._selector]
            else:
                return result
        elif exec_flag:
            raise result
        else:
            return self.result()

    def done(self) -> bool:
        """
        Check if the future item is done.

        Returns:
            bool: True if the future item is done, False otherwise.

        """
        return get_output(file_name=self._file_name)[0]


def execute_tasks_h5(
    future_queue: queue.Queue,
    execute_function: Callable,
    executor_kwargs: dict,
    terminate_function: Optional[Callable] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    disable_dependencies: bool = False,
    pmi_mode: Optional[str] = None,
    wait: bool = True,
    refresh_rate: float = 0.01,
) -> None:
    """
    Execute tasks stored in a queue using HDF5 files.

    Args:
        future_queue (queue.Queue): The queue containing the tasks.
        executor_kwargs (dict): A dictionary of executor arguments required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - cwd (str/None): current working directory where the parallel python task is executed
        execute_function (Callable): The function to execute the tasks.
        terminate_function (Callable): The function to terminate the tasks.
        pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
        backend (str, optional): name of the backend used to spawn tasks.
        disable_dependencies (boolean): Disable resolving future objects during the submission.
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)
        wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
        refresh_rate (float): The rate at which to refresh the result. Defaults to 0.01.

    Returns:
        None

    """
    memory_dict: dict = {}
    process_dict: dict = {}
    cache_dir_dict: dict = {}
    file_name_dict: dict = {}
    duplicate_dict: dict = {}
    while True:
        task_dict = None
        with contextlib.suppress(queue.Empty):
            task_dict = future_queue.get_nowait()
        if task_dict is not None and "shutdown" in task_dict and task_dict["shutdown"]:
            _shutdown_executor(
                wait=wait and task_dict["wait"],
                cancel_futures=task_dict.get("cancel_futures", False),
                memory_dict=memory_dict,
                process_dict=process_dict,
                duplicate_dict=duplicate_dict,
                cache_dir_dict=cache_dir_dict,
                terminate_function=terminate_function,
                pysqa_config_directory=pysqa_config_directory,
                backend=backend,
                refresh_rate=refresh_rate,
            )
            future_queue.task_done()
            future_queue.join()
            break
        elif task_dict is not None:
            task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
                task_dict=task_dict,
                memory_dict=memory_dict,
                file_name_dict=file_name_dict,
            )
            task_resource_dict, cache_key, cache_directory, error_log_file = (
                _get_task_input(
                    task_resource_dict=task_dict["resource_dict"].copy(),
                    executor_kwargs=executor_kwargs,
                )
            )
            task_key, data_dict = serialize_funct(
                fn=task_dict["fn"],
                fn_args=task_args,
                fn_kwargs=task_kwargs,
                resource_dict=task_resource_dict,
                cache_key=cache_key,
            )
            data_dict["error_log_file"] = error_log_file
            if task_key not in memory_dict:
                if os.path.join(
                    cache_directory, task_key + "_o.h5"
                ) not in get_cache_files(cache_directory=cache_directory):
                    file_name = os.path.join(cache_directory, task_key + "_i.h5")
                    if not disable_dependencies:
                        task_dependent_lst = [
                            process_dict[k]
                            for k in future_wait_key_lst
                            if k in process_dict
                        ]
                    else:
                        if len(future_wait_key_lst) > 0:
                            task_dict["future"].set_exception(
                                ValueError(
                                    "Future objects are not supported as input if disable_dependencies=True."
                                )
                            )
                        task_dependent_lst = []
                    process_dict[task_key] = execute_function(
                        command=get_cache_execute_command(
                            file_name=file_name,
                            cores=task_resource_dict["cores"],
                            backend=backend,
                            exclusive=task_resource_dict.get("exclusive", False),
                            openmpi_oversubscribe=task_resource_dict.get(
                                "openmpi_oversubscribe", False
                            ),
                            pmi_mode=pmi_mode,
                        ),
                        file_name=file_name,
                        data_dict=data_dict,
                        task_dependent_lst=task_dependent_lst,
                        resource_dict=task_resource_dict,
                        config_directory=pysqa_config_directory,
                        backend=backend,
                        cache_directory=cache_directory,
                    )
                file_name = os.path.join(cache_directory, task_key + "_o.h5")
                file_name_dict[task_key] = file_name
                queue_id = get_queue_id(file_name=file_name)
                if queue_id is not None:
                    process_dict[task_key] = queue_id
                memory_dict[task_key] = task_dict["future"]
                cache_dir_dict[task_key] = cache_directory
            elif memory_dict[task_key] != task_dict["future"]:
                if task_key not in duplicate_dict:
                    duplicate_dict[task_key] = []
                duplicate_dict[task_key].append(task_dict["future"])
            future_queue.task_done()
        else:
            memory_dict = _refresh_memory_dict(
                memory_dict=memory_dict,
                cache_dir_dict=cache_dir_dict,
                process_dict=process_dict,
                duplicate_dict=duplicate_dict,
                terminate_function=terminate_function,
                pysqa_config_directory=pysqa_config_directory,
                backend=backend,
                refresh_rate=refresh_rate,
            )


def _check_task_output(
    task_key: str,
    future_obj: Future,
    cache_directory: str,
    duplicate_dict: Optional[dict] = None,
) -> Future:
    """
    Check the output of a task and set the result of the future object if available.

    Args:
        task_key (str): The key of the task.
        future_obj (Future): The future object associated with the task.
        cache_directory (str): The directory where the HDF5 files are stored.
        duplicate_dict (dict): The dictionary mapping task keys to their associated duplicate future objects.
    Returns:
        Future: The updated future object.

    """
    file_name = os.path.join(cache_directory, task_key + "_o.h5")
    if not os.path.exists(file_name):
        return future_obj
    exec_flag, no_error_flag, result = get_output(file_name=file_name)
    _update_future(
        future_obj=future_obj,
        exec_flag=exec_flag,
        no_error_flag=no_error_flag,
        result=result,
    )
    if duplicate_dict is not None and task_key in duplicate_dict:
        for duplicate_future in duplicate_dict[task_key]:
            _update_future(
                future_obj=duplicate_future,
                exec_flag=exec_flag,
                no_error_flag=no_error_flag,
                result=result,
            )
        del duplicate_dict[task_key]
    return future_obj


def _update_future(
    future_obj: Future, exec_flag: bool, no_error_flag: bool, result: Any
) -> None:
    """
    Update the future object with the result of the task execution.

    Args:
        future_obj (Future): The future object to be updated.
        exec_flag (bool): Flag indicating whether the task has been executed.
        no_error_flag (bool): Flag indicating whether the task execution resulted in an error.
        result (Any): The result of the task execution.
    """
    if exec_flag and no_error_flag:
        future_obj.set_result(result)
    elif exec_flag:
        future_obj.set_exception(result)


def _convert_args_and_kwargs(
    task_dict: dict, memory_dict: dict, file_name_dict: dict
) -> tuple[list, dict, list]:
    """
    Convert the arguments and keyword arguments in a task dictionary to the appropriate types.

    Args:
        task_dict (dict): The task dictionary containing the arguments and keyword arguments.
        memory_dict (dict): The dictionary mapping future objects to their associated task keys.
        file_name_dict (dict): The dictionary mapping task keys to their corresponding file names.

    Returns:
        Tuple[list, dict, list]: A tuple containing the converted arguments, converted keyword arguments, and a list of future wait keys.

    """
    task_args = []
    task_kwargs = {}
    future_wait_key_lst = []
    for arg in task_dict["args"]:
        selector = None
        if isinstance(arg, Future):
            if hasattr(arg, "_future") and hasattr(arg, "_selector"):
                selector = arg._selector
                future = arg._future
            else:
                future = arg
            match_found = False
            for k, v in memory_dict.items():
                if future == v:
                    task_args.append(
                        FutureItem(file_name=file_name_dict[k], selector=selector)
                    )
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_args.append(future.result())
        else:
            task_args.append(arg)
    for key, arg in task_dict["kwargs"].items():
        selector = None
        if isinstance(arg, Future):
            if hasattr(arg, "_future") and hasattr(arg, "_selector"):
                selector = arg._selector
                future = arg._future
            else:
                future = arg
            match_found = False
            for k, v in memory_dict.items():
                if future == v:
                    task_kwargs[key] = FutureItem(
                        file_name=file_name_dict[k], selector=selector
                    )
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_kwargs[key] = future.result()
        else:
            task_kwargs[key] = arg
    return task_args, task_kwargs, future_wait_key_lst


def _refresh_memory_dict(
    memory_dict: dict,
    cache_dir_dict: dict,
    process_dict: dict,
    duplicate_dict: Optional[dict] = None,
    terminate_function: Optional[Callable] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    refresh_rate: float = 0.01,
) -> dict:
    """
    Refresh memory dictionary

    Args:
        memory_dict (dict): dictionary with task keys and future objects
        cache_dir_dict (dict): dictionary with task keys and cache directories
        process_dict (dict): dictionary with task keys and process reference.
        duplicate_dict (dict): dictionary with task keys and duplicate future objects.
        terminate_function (callable): The function to terminate the tasks.
        pysqa_config_directory (str): path to the pysqa config directory (only for pysqa based backend).
        backend (str): name of the backend used to spawn tasks.
        refresh_rate (float): The rate at which to refresh the result. Defaults to 0.01.

    Returns:
        dict: Updated memory dictionary
    """
    cancelled_lst = [
        key for key, value in memory_dict.items() if value.done() and value.cancelled()
    ]
    _cancel_processes(
        process_dict={k: v for k, v in process_dict.items() if k in cancelled_lst},
        terminate_function=terminate_function,
        pysqa_config_directory=pysqa_config_directory,
        backend=backend,
    )
    memory_updated_dict = {
        key: _check_task_output(
            task_key=key,
            future_obj=value,
            cache_directory=cache_dir_dict[key],
            duplicate_dict=duplicate_dict,
        )
        for key, value in memory_dict.items()
        if not value.done()
    }
    if len(memory_updated_dict) == len(memory_dict):
        sleep(refresh_rate)
    return memory_updated_dict


def _cancel_processes(
    process_dict: dict,
    terminate_function: Optional[Callable] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Cancel processes

    Args:
        process_dict (dict): dictionary with task keys and process reference.
        terminate_function (callable): The function to terminate the tasks.
        pysqa_config_directory (str): path to the pysqa config directory (only for pysqa based backend).
        backend (str): name of the backend used to spawn tasks.
    """
    if terminate_function is not None and terminate_function == subprocess_terminate:
        for task in process_dict.values():
            terminate_function(task=task)
    elif terminate_function is not None and backend is not None:
        for queue_id in process_dict.values():
            terminate_function(
                queue_id=queue_id,
                config_directory=pysqa_config_directory,
                backend=backend,
            )


def _get_task_input(
    task_resource_dict: dict, executor_kwargs: dict
) -> tuple[dict, Optional[str], str, Optional[str]]:
    """
    Merge per-task resource requirements with executor defaults and extract scheduling metadata.

    Executor-level kwargs fill in any keys not already present in the per-task resource dict.
    The special keys ``cache_key``, ``cache_directory``, and ``error_log_file`` are popped from
    the merged dict and returned separately so callers do not forward them to the backend.

    Args:
        task_resource_dict (dict): Per-task resource dict from the submitted future, modified in place.
        executor_kwargs (dict): Executor-level defaults (e.g. cores, cwd, cache_directory).

    Returns:
        Tuple[dict, Optional[str], str, Optional[str]]:
            - merged resource dict (without scheduling-only keys)
            - cache_key (str or None)
            - cache_directory (str, absolute path)
            - error_log_file (str or None)
    """
    task_resource_dict.update(
        {k: v for k, v in executor_kwargs.items() if k not in task_resource_dict}
    )
    cache_key = task_resource_dict.pop("cache_key", None)
    cache_directory = os.path.abspath(task_resource_dict.pop("cache_directory"))
    error_log_file = task_resource_dict.pop("error_log_file", None)
    return task_resource_dict, cache_key, cache_directory, error_log_file


def _cancel_futures(future_dict: dict):
    """
    Cancel all pending futures in the dictionary.

    Args:
        future_dict (dict): Mapping of task keys to Future objects. Already-done futures are
            skipped; pending ones are cancelled.
    """
    for value in future_dict.values():
        if not value.done():
            value.cancel()


def _shutdown_executor(
    wait: bool,
    cancel_futures: bool,
    memory_dict: dict,
    process_dict: dict,
    cache_dir_dict: dict,
    duplicate_dict: Optional[dict] = None,
    terminate_function: Optional[Callable] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    refresh_rate: float = 0.01,
):
    """
    Shut down the file-based executor, optionally waiting for or cancelling pending tasks.

    The four combinations of ``wait`` / ``cancel_futures`` mirror the semantics of
    concurrent.futures.Executor.shutdown():

    * wait=True, cancel_futures=False  – poll until all tasks finish.
    * wait=True, cancel_futures=True   – cancel pending futures then wait for running ones.
    * wait=False, cancel_futures=True  – cancel everything immediately without blocking.
    * wait=False, cancel_futures=False – detach tasks and mark futures cancelled.

    Args:
        wait (bool): Whether to block until all outstanding tasks have resolved.
        cancel_futures (bool): Whether to cancel futures that have not yet started.
        memory_dict (dict): Mapping of task keys to their Future objects.
        process_dict (dict): Mapping of task keys to process handles or queue IDs.
        duplicate_dict (dict): Mapping of task keys to lists of duplicate Future objects.
        cache_dir_dict (dict): Mapping of task keys to the cache directory for each task.
        terminate_function (Callable, optional): Function used to terminate running processes.
        pysqa_config_directory (str, optional): Path to the pysqa config directory.
        backend (str, optional): Name of the backend ("slurm", "flux", or None for subprocess).
        refresh_rate (float): Polling interval in seconds when waiting for tasks. Defaults to 0.01.
    """
    if wait and not cancel_futures:
        while len(memory_dict) > 0:
            memory_dict = _refresh_memory_dict(
                memory_dict=memory_dict,
                cache_dir_dict=cache_dir_dict,
                process_dict=process_dict,
                duplicate_dict=duplicate_dict,
                terminate_function=terminate_function,
                pysqa_config_directory=pysqa_config_directory,
                backend=backend,
                refresh_rate=refresh_rate,
            )
    elif wait and cancel_futures:
        for value in memory_dict.values():
            if not value.done():
                value.cancel()
        while len(memory_dict) > 0:
            memory_dict = _refresh_memory_dict(
                memory_dict=memory_dict,
                cache_dir_dict=cache_dir_dict,
                process_dict=process_dict,
                duplicate_dict=duplicate_dict,
                terminate_function=terminate_function,
                pysqa_config_directory=pysqa_config_directory,
                backend=backend,
                refresh_rate=refresh_rate,
            )
    elif cancel_futures:  # wait is False
        _cancel_processes(
            process_dict=process_dict,
            terminate_function=terminate_function,
            pysqa_config_directory=pysqa_config_directory,
            backend=backend,
        )
        _cancel_futures(future_dict=memory_dict)
    else:  # wait is False and cancel_futures is False
        memory_dict = _refresh_memory_dict(
            memory_dict=memory_dict,
            cache_dir_dict=cache_dir_dict,
            process_dict=process_dict,
            duplicate_dict=duplicate_dict,
            terminate_function=terminate_function,
            pysqa_config_directory=pysqa_config_directory,
            backend=backend,
            refresh_rate=refresh_rate,
        )
        # The future objects are detached so mark them as cancelled even though the processes are
        # not terminated. This is to prevent the main process from waiting indefinitely for the results.
        _cancel_futures(future_dict=memory_dict)
