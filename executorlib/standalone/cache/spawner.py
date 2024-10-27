import subprocess
import time
from typing import Optional


def execute_in_subprocess(
    command: list,
    task_dependent_lst: list = [],
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """
    Execute a command in a subprocess.

    Args:
        command (list): The command to be executed.
        task_dependent_lst (list): A list of subprocesses that the current subprocess depends on. Defaults to [].
        cwd (str/None): current working directory where the parallel python task is executed

    Returns:
        subprocess.Popen: The subprocess object.

    """
    while len(task_dependent_lst) > 0:
        task_dependent_lst = [
            task for task in task_dependent_lst if task.poll() is None
        ]
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
