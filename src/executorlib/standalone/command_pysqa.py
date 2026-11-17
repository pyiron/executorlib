import contextlib
import os
import subprocess
from time import monotonic

from pysqa import QueueAdapter

# Minimum time between two queries of the queuing system for the status of a task whose output
# file has not appeared yet. Detecting a dead job (timeout, OOM, node failure, scancel, ...) relies
# on this status query, but it must not be issued on every poll of the (much faster) refresh_rate
# loop, as that would flood the queuing system commands (e.g. squeue/sacct) with requests.
_JOB_STATUS_CHECK_INTERVAL = 30.0
# Time to wait for the output file of a job, which the queuing system reports as successfully
# finished, to become visible on a shared filesystem before the task is failed.
_JOB_OUTPUT_GRACE_PERIOD = 60.0


def pysqa_job_output_validation(
    task_key: str,
    file_name: str,
    queue_id: int,
    status_check_dict: dict | None,
    pysqa_config_directory: str | None = None,
    backend: str | None = None,
    job_status_check_interval: float = _JOB_STATUS_CHECK_INTERVAL,
) -> bool:
    """
    Check whether the queuing system job backing a task has died without ever writing its output
    file. Jobs which the queuing system reports as successfully finished are given a grace period
    for their output to appear on a shared filesystem. Status checks are throttled to at most once
    every ``job_status_check_interval`` seconds per task, to avoid flooding the queuing system on
    every poll of the refresh loop.

    A dead job is recognized in two ways, since queuing systems differ in whether they drop
    terminated jobs from their listing: flux's "flux jobs -a" keeps listing inactive jobs, while for
    slurm pysqa falls back to the accounting database (sacct) once a job left squeue. Both report
    pysqa's terminal-failure status "error" for jobs which failed (timeout, OOM, node failure,
    scancel, ...). A job which is unknown to the queuing system (status None), for example because
    slurm accounting is not available, is treated as dead as well.

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
    check_state = (
        status_check_dict.get(task_key, {}) if status_check_dict is not None else {}
    )
    if isinstance(check_state, dict):
        last_checked = check_state.get("last_checked", 0.0)
        completed_at = check_state.get("completed_at")
    else:
        last_checked = check_state
        completed_at = None
    if completed_at is not None:
        return now - completed_at >= _JOB_OUTPUT_GRACE_PERIOD
    if now - last_checked < job_status_check_interval:
        return False
    if status_check_dict is not None:
        status_check_dict[task_key] = {"last_checked": now}
    status = pysqa_get_status_of_job(
        queue_id=queue_id,
        config_directory=pysqa_config_directory,
        backend=backend,
    )
    if status == "finished":
        if status_check_dict is None:
            return not os.path.exists(file_name)
        status_check_dict[task_key] = {"last_checked": now, "completed_at": now}
        return False
    return (status is None or status == "error") and not os.path.exists(file_name)


def pysqa_terminate(
    queue_id: int,
    config_directory: str | None = None,
    backend: str | None = None,
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
        execute_command=_pysqa_execute_status_command,
    )
    status = qa.get_status_of_job(process_id=queue_id)
    if status is not None and status not in ["finished", "error"]:
        with contextlib.suppress(subprocess.CalledProcessError):
            qa.delete_job(process_id=queue_id)


def pysqa_get_status_of_job(
    queue_id: int,
    config_directory: str | None = None,
    backend: str | None = None,
) -> str | None:
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
        execute_command=_pysqa_execute_status_command,
    )
    return qa.get_status_of_job(process_id=queue_id)


def pysqa_execute_command(
    commands: str | list[str],
    working_directory: str | None = None,
    split_output: bool = True,
    shell: bool = False,
    error_filename: str = "pysqa.err",
) -> str | list[str]:
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


def _pysqa_execute_status_command(
    commands: str | list[str],
    working_directory: str | None = None,
    split_output: bool = True,
    shell: bool = False,
    error_filename: str = "pysqa.err",
) -> str | list[str] | None:
    """
    Variant of pysqa_execute_command for job status queries. pysqa falls back to the slurm
    accounting database (sacct) for jobs which are no longer listed by squeue. On clusters without
    slurm accounting sacct fails, in this case None is returned so pysqa reports the job as unknown
    rather than raising an exception.
    """
    try:
        return pysqa_execute_command(
            commands=commands,
            working_directory=working_directory,
            split_output=split_output,
            shell=shell,
            error_filename=error_filename,
        )
    except subprocess.CalledProcessError:
        if "sacct" in commands:
            return None
        raise
