import contextlib
import os
import subprocess
from time import monotonic
from typing import Optional, Union

from pysqa import QueueAdapter


def pysqa_job_output_validation(
    task_key: str,
    file_name: str,
    queue_id: int,
    status_check_dict: Optional[dict],
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    job_status_check_interval: float = 30.0
) -> bool:
    """
    Check whether the queuing system job backing a task has died without ever writing its output
    file. Only applies to queuing system backends (pysqa) and is throttled to at most once every
    ``job_status_check_interval`` seconds per task, to avoid flooding the queuing system with
    status queries on every poll of the (much faster) refresh_rate loop.

    A dead job is recognized in two ways, since queuing systems differ in whether they drop
    terminated jobs from their listing: slurm's squeue removes a job as soon as it is gone
    (status None), while flux's "flux jobs -a" keeps listing inactive jobs and instead reports
    pysqa's terminal-failure status "error" (the same status pysqa_terminate already treats as
    not alive).

    Args:
        task_key (str): The key of the task.
        file_name (str): Path of the expected output HDF5 file.
        queue_id (int, optional): The queuing system ID of the task.
        pysqa_config_directory (str, optional): path to the pysqa config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
        status_check_dict (dict): Dictionary tracking when each task's job status was last queried.
        job_status_check_interval (float): Minimum time interval between job status checks for the same task.

    Returns:
        bool: True if the job is no longer known to the queuing system, or is reported as having
            errored out, and still has no output.
    """
    now = monotonic()
    last_checked = (
        status_check_dict.get(task_key, 0.0) if status_check_dict is not None else 0.0
    )
    if now - last_checked < job_status_check_interval:
        return False
    if status_check_dict is not None:
        status_check_dict[task_key] = now
    status = pysqa_get_status_of_job(
        queue_id=queue_id,
        config_directory=pysqa_config_directory,
        backend=backend,
    )
    return (status is None or status == "error") and not os.path.exists(file_name)


def pysqa_terminate(
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


def pysqa_get_status_of_job(
    queue_id: int,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
) -> Optional[str]:
    """
    Query the status of a job from the queuing system.

    Args:
        queue_id (int): Queuing system ID of the job to check.
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].

    Returns:
        str: status of the job as reported by the queuing system, None if the queuing system no
             longer knows about the job (e.g. it timed out, was killed or already finished).
    """
    qa = QueueAdapter(
        directory=config_directory,
        queue_type=backend,
        execute_command=pysqa_execute_command,
    )
    return qa.get_status_of_job(process_id=queue_id)


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
