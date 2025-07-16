import contextlib
import os
import subprocess
from typing import Optional, Union

from pysqa import QueueAdapter

from executorlib.standalone.inputcheck import check_file_exists
from executorlib.task_scheduler.file.hdf import dump, get_queue_id


def execute_with_pysqa(
    command: list,
    file_name: str,
    data_dict: dict,
    cache_directory: str,
    task_dependent_lst: Optional[list[int]] = None,
    resource_dict: Optional[dict] = None,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
) -> Optional[int]:
    """
    Execute a command by submitting it to the queuing system

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
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].

    Returns:
        int: queuing system ID
    """
    if task_dependent_lst is None:
        task_dependent_lst = []
    qa = QueueAdapter(
        directory=config_directory,
        queue_type=backend,
        execute_command=_pysqa_execute_command,
    )
    queue_id = get_queue_id(file_name=file_name)
    if os.path.exists(file_name) and (
        queue_id is None or qa.get_status_of_job(process_id=queue_id) is None
    ):
        os.remove(file_name)
        dump(file_name=file_name, data_dict=data_dict)
    elif not os.path.exists(file_name):
        dump(file_name=file_name, data_dict=data_dict)
    check_file_exists(file_name=file_name)
    if queue_id is None or qa.get_status_of_job(process_id=queue_id) is None:
        if resource_dict is None:
            resource_dict = {}
        if "cwd" in resource_dict and resource_dict["cwd"] is not None:
            cwd = resource_dict["cwd"]
        else:
            folder = command[-1].split("_i.h5")[0]
            cwd = os.path.join(cache_directory, folder)
            os.makedirs(cwd, exist_ok=True)
        submit_kwargs = {
            "command": " ".join(command),
            "dependency_list": [str(qid) for qid in task_dependent_lst],
            "working_directory": os.path.abspath(cwd),
        }
        if "cwd" in resource_dict:
            del resource_dict["cwd"]
        if "threads_per_core" in resource_dict:
            resource_dict["cores"] *= resource_dict["threads_per_core"]
            del resource_dict["threads_per_core"]
        unsupported_keys = [
            "gpus_per_core",
            "openmpi_oversubscribe",
            "slurm_cmd_args",
        ]
        for k in unsupported_keys:
            if k in resource_dict:
                del resource_dict[k]
        if "job_name" not in resource_dict:
            resource_dict["job_name"] = os.path.basename(
                os.path.dirname(os.path.abspath(cwd))
            )
        submit_kwargs.update(resource_dict)
        queue_id = qa.submit_job(**submit_kwargs)
        dump(file_name=file_name, data_dict={"queue_id": queue_id})
    return queue_id


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
        execute_command=_pysqa_execute_command,
    )
    status = qa.get_status_of_job(process_id=queue_id)
    if status is not None and status not in ["finished", "error"]:
        with contextlib.suppress(subprocess.CalledProcessError):
            qa.delete_job(process_id=queue_id)


def terminate_tasks_in_cache(
    cache_directory: str,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Delete all jobs stored in the cache directory from the queuing system

    Args:
        cache_directory (str): The directory to store cache files.
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
    """
    hdf5_file_lst = []
    for root, _, files in os.walk(cache_directory):
        hdf5_file_lst += [os.path.join(root, f) for f in files if f[-5:] == "_i.h5"]

    for f in hdf5_file_lst:
        queue_id = get_queue_id(f)
        if queue_id is not None:
            terminate_with_pysqa(
                queue_id=queue_id,
                config_directory=config_directory,
                backend=backend,
            )


def _pysqa_execute_command(
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
