import hashlib
import importlib.util
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import Future
from typing import Any, Tuple

import cloudpickle

from executorlib.cache.hdf import dump, get_output, load
from executorlib.shared.executor import get_command_path


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


def backend_load_file(file_name: str) -> dict:
    """
    Load the data from an HDF5 file and convert FutureItem objects to their results.

    Args:
        file_name (str): The name of the HDF5 file.

    Returns:
        dict: The loaded data from the file.

    """
    apply_dict = load(file_name=file_name)
    apply_dict["args"] = [
        arg if not isinstance(arg, FutureItem) else arg.result()
        for arg in apply_dict["args"]
    ]
    apply_dict["kwargs"] = {
        key: arg if not isinstance(arg, FutureItem) else arg.result()
        for key, arg in apply_dict["kwargs"].items()
    }
    return apply_dict


def backend_write_file(file_name: str, output: Any) -> None:
    """
    Write the output to an HDF5 file.

    Args:
        file_name (str): The name of the HDF5 file.
        output (Any): The output to be written.

    Returns:
        None

    """
    file_name_out = os.path.splitext(file_name)[0]
    os.rename(file_name, file_name_out + ".h5ready")
    dump(file_name=file_name_out + ".h5ready", data_dict={"output": output})
    os.rename(file_name_out + ".h5ready", file_name_out + ".h5out")


def execute_in_subprocess(
    command: list, task_dependent_lst: list = []
) -> subprocess.Popen:
    """
    Execute a command in a subprocess.

    Args:
        command (list): The command to be executed.
        task_dependent_lst (list, optional): A list of subprocesses that the current subprocess depends on. Defaults to [].

    Returns:
        subprocess.Popen: The subprocess object.

    """
    while len(task_dependent_lst) > 0:
        task_dependent_lst = [
            task for task in task_dependent_lst if task.poll() is None
        ]
    return subprocess.Popen(command, universal_newlines=True)


def execute_tasks_h5(
    future_queue: queue.Queue,
    cache_directory: str,
    cores_per_worker: int,
    execute_function: callable,
) -> None:
    """
    Execute tasks stored in a queue using HDF5 files.

    Args:
        future_queue (queue.Queue): The queue containing the tasks.
        cache_directory (str): The directory to store the HDF5 files.
        cores_per_worker (int): The number of cores per worker.
        execute_function (callable): The function to execute the tasks.

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
            future_queue.task_done()
            future_queue.join()
            break
        elif task_dict is not None:
            task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
                task_dict=task_dict,
                memory_dict=memory_dict,
                file_name_dict=file_name_dict,
            )
            task_key, data_dict = _serialize_funct_h5(
                task_dict["fn"], *task_args, **task_kwargs
            )
            if task_key not in memory_dict.keys():
                if task_key + ".h5out" not in os.listdir(cache_directory):
                    file_name = os.path.join(cache_directory, task_key + ".h5in")
                    dump(file_name=file_name, data_dict=data_dict)
                    process_dict[task_key] = execute_function(
                        command=_get_execute_command(
                            file_name=file_name,
                            cores=cores_per_worker,
                        ),
                        task_dependent_lst=[
                            process_dict[k] for k in future_wait_key_lst
                        ],
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


def execute_task_in_file(file_name: str) -> None:
    """
    Execute the task stored in a given HDF5 file.

    Args:
        file_name (str): The file name of the HDF5 file as an absolute path.

    Returns:
        None
    """
    apply_dict = backend_load_file(file_name=file_name)
    result = apply_dict["fn"].__call__(*apply_dict["args"], **apply_dict["kwargs"])
    backend_write_file(
        file_name=file_name,
        output=result,
    )


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


def _get_hash(binary: bytes) -> str:
    """
    Get the hash of a binary.

    Args:
        binary (bytes): The binary to be hashed.

    Returns:
        str: The hash of the binary.

    """
    # Remove specification of jupyter kernel from hash to be deterministic
    binary_no_ipykernel = re.sub(b"(?<=/ipykernel_)(.*)(?=/)", b"", binary)
    return str(hashlib.md5(binary_no_ipykernel).hexdigest())


def _serialize_funct_h5(fn: callable, *args: Any, **kwargs: Any) -> Tuple[str, dict]:
    """
    Serialize a function and its arguments and keyword arguments into an HDF5 file.

    Args:
        fn (callable): The function to be serialized.
        *args (Any): The arguments of the function.
        **kwargs (Any): The keyword arguments of the function.

    Returns:
        Tuple[str, dict]: A tuple containing the task key and the serialized data.

    """
    binary_all = cloudpickle.dumps({"fn": fn, "args": args, "kwargs": kwargs})
    task_key = fn.__name__ + _get_hash(binary=binary_all)
    data = {"fn": fn, "args": args, "kwargs": kwargs}
    return task_key, data


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
