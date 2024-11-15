import importlib.util
import os
import queue
import sys
from concurrent.futures import Future
from typing import Optional, Tuple

from executorlib.standalone.command import get_command_path
from executorlib.standalone.hdf import dump, get_output
from executorlib.standalone.serialize import serialize_funct_h5


class FutureItem:
    def __init__(self, file_name: str):
        """
        Initialize a FutureItem object.

        Args:
            file_name (str): The name of the file.

        """
        self._file_name = file_name

    def result(self) -> str:
        """
        Get the result of the future item.

        Returns:
            str: The result of the future item.

        """
        exec_flag, result = get_output(file_name=self._file_name)
        if exec_flag:
            return result
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
    cache_directory: str,
    execute_function: callable,
    resource_dict: dict,
    terminate_function: Optional[callable] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    disable_dependencies: bool = False,
) -> None:
    """
    Execute tasks stored in a queue using HDF5 files.

    Args:
        future_queue (queue.Queue): The queue containing the tasks.
        cache_directory (str): The directory to store the HDF5 files.
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - cwd (str/None): current working directory where the parallel python task is executed
        execute_function (callable): The function to execute the tasks.
        terminate_function (callable): The function to terminate the tasks.
        pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
        backend (str, optional): name of the backend used to spawn tasks.

    Returns:
        None

    """
    memory_dict, process_dict, file_name_dict = {}, {}, {}
    while True:
        task_dict = None
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            pass
        if (
            task_dict is not None
            and "shutdown" in task_dict.keys()
            and task_dict["shutdown"]
        ):
            if terminate_function is not None:
                for task in process_dict.values():
                    terminate_function(task=task)
            future_queue.task_done()
            future_queue.join()
            break
        elif task_dict is not None:
            task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
                task_dict=task_dict,
                memory_dict=memory_dict,
                file_name_dict=file_name_dict,
            )
            task_resource_dict = task_dict["resource_dict"].copy()
            task_resource_dict.update(
                {k: v for k, v in resource_dict.items() if k not in task_resource_dict}
            )
            task_key, data_dict = serialize_funct_h5(
                fn=task_dict["fn"],
                fn_args=task_args,
                fn_kwargs=task_kwargs,
                resource_dict=task_resource_dict,
            )
            if task_key not in memory_dict.keys():
                if task_key + ".h5out" not in os.listdir(cache_directory):
                    file_name = os.path.join(cache_directory, task_key + ".h5in")
                    dump(file_name=file_name, data_dict=data_dict)
                    if not disable_dependencies:
                        task_dependent_lst = [
                            process_dict[k] for k in future_wait_key_lst
                        ]
                    else:
                        if len(future_wait_key_lst) > 0:
                            raise ValueError(
                                "Future objects are not supported as input if disable_dependencies=True."
                            )
                        task_dependent_lst = []
                    process_dict[task_key] = execute_function(
                        command=_get_execute_command(
                            file_name=file_name,
                            cores=task_resource_dict["cores"],
                        ),
                        file_name=file_name,
                        task_dependent_lst=task_dependent_lst,
                        resource_dict=task_resource_dict,
                        config_directory=pysqa_config_directory,
                        backend=backend,
                        cache_directory=cache_directory,
                    )
                file_name_dict[task_key] = os.path.join(
                    cache_directory, task_key + ".h5out"
                )
                memory_dict[task_key] = task_dict["future"]
            future_queue.task_done()
        else:
            memory_dict = {
                key: _check_task_output(
                    task_key=key, future_obj=value, cache_directory=cache_directory
                )
                for key, value in memory_dict.items()
                if not value.done()
            }


def _get_execute_command(file_name: str, cores: int = 1) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        file_name (str): The name of the file.
        cores (int, optional): Number of cores used to execute the task. Defaults to 1.

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1 and importlib.util.find_spec("mpi4py") is not None:
        command_lst = (
            ["mpiexec", "-n", str(cores)]
            + command_lst
            + [get_command_path(executable="cache_parallel.py"), file_name]
        )
    elif cores > 1:
        raise ImportError(
            "mpi4py is required for parallel calculations. Please install mpi4py."
        )
    else:
        command_lst += [get_command_path(executable="cache_serial.py"), file_name]
    return command_lst


def _check_task_output(
    task_key: str, future_obj: Future, cache_directory: str
) -> Future:
    """
    Check the output of a task and set the result of the future object if available.

    Args:
        task_key (str): The key of the task.
        future_obj (Future): The future object associated with the task.
        cache_directory (str): The directory where the HDF5 files are stored.

    Returns:
        Future: The updated future object.

    """
    file_name = os.path.join(cache_directory, task_key + ".h5out")
    if not os.path.exists(file_name):
        return future_obj
    exec_flag, result = get_output(file_name=file_name)
    if exec_flag:
        future_obj.set_result(result)
    return future_obj


def _convert_args_and_kwargs(
    task_dict: dict, memory_dict: dict, file_name_dict: dict
) -> Tuple[list, dict, list]:
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
        if isinstance(arg, Future):
            match_found = False
            for k, v in memory_dict.items():
                if arg == v:
                    task_args.append(FutureItem(file_name=file_name_dict[k]))
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_args.append(arg.result())
        else:
            task_args.append(arg)
    for key, arg in task_dict["kwargs"].items():
        if isinstance(arg, Future):
            match_found = False
            for k, v in memory_dict.items():
                if arg == v:
                    task_kwargs[key] = FutureItem(file_name=file_name_dict[k])
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_kwargs[key] = arg.result()
        else:
            task_kwargs[key] = arg
    return task_args, task_kwargs, future_wait_key_lst
