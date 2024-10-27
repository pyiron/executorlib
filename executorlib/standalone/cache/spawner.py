import subprocess
import time
from typing import Optional


def execute_in_subprocess(
    command: list,
    task_dependent_lst: list = [],
    resource_dict: Optional[dict] = None,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
) -> subprocess.Popen:
    """
    Execute a command in a subprocess.

    Args:
        command (list): The command to be executed.
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
        resource_dict = {"cwd": None}
    elif len(resource_dict) == 0:
        resource_dict = {"cwd": None}
    return subprocess.Popen(command, universal_newlines=True, cwd=resource_dict["cwd"])


def terminate_subprocess(task):
    """
    Terminate a subprocess and wait for it to complete.

    Args:
        task (subprocess.Popen): The subprocess.Popen instance to terminate
    """
    task.terminate()
    while task.poll() is None:
        time.sleep(0.1)
