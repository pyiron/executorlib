import contextlib
import subprocess
from typing import Optional, Union

from pysqa import QueueAdapter


def terminate_with_pysqa(
    queue_id: int,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Delete job from queuing system

    Args:
        queue_id (int): Queuing system ID of the job to delete.
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
    """
    qa = QueueAdapter(
        directory=config_directory,
        queue_type=backend,
        execute_command=pysqa_execute_command,
    )
    status = qa.get_status_of_job(process_id=queue_id)
    if status is not None and status not in ["finished", "error"]:
        with contextlib.suppress(subprocess.CalledProcessError):
            qa.delete_job(process_id=queue_id)


def pysqa_execute_command(
    commands: str,
    working_directory: Optional[str] = None,
    split_output: bool = True,
    shell: bool = False,
    error_filename: str = "pysqa.err",
) -> Union[str, list[str]]:
    """
    A wrapper around the subprocess.check_output function. Modified from pysqa to raise an exception if the subprocess
    fails to submit the job to the queue.

    Args:
        commands (str): The command(s) to be executed on the command line
        working_directory (str, optional): The directory where the command is executed. Defaults to None.
        split_output (bool, optional): Boolean flag to split newlines in the output. Defaults to True.
        shell (bool, optional): Additional switch to convert commands to a single string. Defaults to False.
        error_filename (str, optional): In case the execution fails, the output is written to this file. Defaults to "pysqa.err".

    Returns:
        Union[str, List[str]]: Output of the shell command either as a string or as a list of strings
    """
    if shell and isinstance(commands, list):
        commands = " ".join(commands)
    out = subprocess.check_output(
        commands,
        cwd=working_directory,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=not isinstance(commands, list),
    )
    if out is not None and split_output:
        return out.split("\n")
    else:
        return out
