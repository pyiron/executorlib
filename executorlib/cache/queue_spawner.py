import os
import subprocess
from typing import List, Optional, Tuple, Union

from pysqa import QueueAdapter

from executorlib.standalone.hdf import dump, get_queue_id
from executorlib.standalone.inputcheck import check_file_exists


def execute_with_pysqa(
    command: list,
    task_dependent_lst: list[int] = [],
    file_name: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
    cache_directory: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Execute a command by submitting it to the queuing system

    Args:
        command (list): The command to be executed.
        task_dependent_lst (list): A list of subprocesses that the current subprocess depends on. Defaults to [].
        file_name (str): Name of the HDF5 file which contains the Python function
        resource_dict (dict): resource dictionary, which defines the resources used for the execution of the function.
                              Example resource dictionary: {
                                  cwd: None,
                              }
        config_directory (str, optional): path to the config directory.
        backend (str, optional): name of the backend used to spawn tasks.
        cache_directory (str): The directory to store the HDF5 files.

    Returns:
        int: queuing system ID
    """
    check_file_exists(file_name=file_name)
    queue_id = get_queue_id(file_name=file_name)
    qa = QueueAdapter(
        directory=config_directory,
        queue_type=backend,
        execute_command=_pysqa_execute_command,
    )
    if queue_id is None or qa.get_status_of_job(process_id=queue_id) is None:
        if resource_dict is None:
            resource_dict = {}
        if "cwd" in resource_dict and resource_dict["cwd"] is not None:
            cwd = resource_dict["cwd"]
        else:
            cwd = cache_directory
        submit_kwargs = {
            "command": " ".join(command),
            "dependency_list": [str(qid) for qid in task_dependent_lst],
            "working_directory": os.path.abspath(cwd),
        }
        if "cwd" in resource_dict:
            del resource_dict["cwd"]
        unsupported_keys = [
            "threads_per_core",
            "gpus_per_core",
            "openmpi_oversubscribe",
            "slurm_cmd_args",
        ]
        for k in unsupported_keys:
            if k in resource_dict:
                del resource_dict[k]
        if "job_name" not in resource_dict:
            resource_dict["job_name"] = "pysqa"
        submit_kwargs.update(resource_dict)
        queue_id = qa.submit_job(**submit_kwargs)
        dump(file_name=file_name, data_dict={"queue_id": queue_id})
    return queue_id


def _pysqa_execute_command(
    commands: str,
    working_directory: Optional[str] = None,
    split_output: bool = True,
    shell: bool = False,
    error_filename: str = "pysqa.err",
) -> Union[str, List[str]]:
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
