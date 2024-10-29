import os
from typing import List, Optional

from pysqa import QueueAdapter


def execute_with_pysqa(
    command: str,
    resource_dict: dict,
    task_dependent_lst: List[int] = [],
    config_directory: Optional[str] = None,
    backend: Optional[str] = None,
) -> int:
    """
    Execute a command by submitting it to the queuing system

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
        int: queuing system ID
    """
    if resource_dict is None:
        resource_dict = {"cwd": "."}
    qa = QueueAdapter(directory=config_directory, queue_type=backend)
    submit_kwargs = {
        "command": " ".join(command),
        "dependency_list": [str(qid) for qid in task_dependent_lst],
        "working_directory": os.path.abspath(resource_dict["cwd"]),
    }
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
    submit_kwargs.update(resource_dict)
    return qa.submit_job(**submit_kwargs)
