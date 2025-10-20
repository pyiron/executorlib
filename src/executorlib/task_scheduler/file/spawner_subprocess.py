import os
import subprocess
import time
from typing import Optional

from executorlib.standalone.hdf import dump
from executorlib.standalone.inputcheck import check_file_exists


def execute_in_subprocess(
    command: list,
    file_name: str,
    data_dict: dict,
    cache_directory: Optional[str] = None,
    task_dependent_lst: Optional[list] = None,
    resource_dict: Optional[dict] = None,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
) -> subprocess.Popen:
    """
    Execute a command in a subprocess.

    Args:
        command (list): The command to be executed.
        file_name (str): Name of the HDF5 file which contains the Python function
        data_dict (dict): dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
        cache_directory (str): The directory to store the HDF5 files.
        task_dependent_lst (list): A list of subprocesses that the current subprocess depends on. Defaults to [].
        resource_dict (dict): resource dictionary, which defines the resources used for the execution of the function.
                              Example resource dictionary: {
                                  cwd: None,
                              }
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks.

    Returns:
        subprocess.Popen: The subprocess object.

    """
    if task_dependent_lst is None:
        task_dependent_lst = []
    if os.path.exists(file_name):
        os.remove(file_name)
    dump(file_name=file_name, data_dict=data_dict)
    check_file_exists(file_name=file_name)
    while len(task_dependent_lst) > 0:
        task_dependent_lst = [
            task for task in task_dependent_lst if task.poll() is None
        ]
    if config_directory is not None:
        raise ValueError(
            "config_directory parameter is not supported for subprocess spawner."
        )
    if backend is not None:
        raise ValueError("backend parameter is not supported for subprocess spawner.")
    if resource_dict is None:
        resource_dict = {}
    cwd = resource_dict.get("cwd", cache_directory)
    if cwd is not None:
        os.makedirs(cwd, exist_ok=True)
    return subprocess.Popen(command, universal_newlines=True, cwd=cwd)


def terminate_subprocess(task):
    """
    Terminate a subprocess and wait for it to complete.

    Args:
        task (subprocess.Popen): The subprocess.Popen instance to terminate
    """
    task.terminate()
    while task.poll() is None:
        time.sleep(0.1)
