import os
from threading import Thread
from typing import Callable, Optional

from executorlib.standalone.inputcheck import (
    check_executor,
    check_flux_executor_pmi_mode,
    check_flux_log_files,
    check_hostname_localhost,
    check_max_workers_and_cores,
    check_nested_flux_executor,
)
from executorlib.task_scheduler.base import TaskSchedulerBase
from executorlib.task_scheduler.file.shared import execute_tasks_h5
from executorlib.task_scheduler.file.subprocess_spawner import (
    execute_in_subprocess,
    terminate_subprocess,
)

try:
    from executorlib.task_scheduler.file.queue_spawner import execute_with_pysqa
except ImportError:
    # If pysqa is not available fall back to executing tasks in a subprocess
    execute_with_pysqa = execute_in_subprocess  # type: ignore


class FileTaskScheduler(TaskSchedulerBase):
    def __init__(
        self,
        cache_directory: str = "executorlib_cache",
        resource_dict: Optional[dict] = None,
        execute_function: Callable = execute_with_pysqa,
        terminate_function: Optional[Callable] = None,
        pysqa_config_directory: Optional[str] = None,
        backend: Optional[str] = None,
        disable_dependencies: bool = False,
    ):
        """
        Initialize the FileExecutor.

        Args:
            cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - cwd (str/None): current working directory where the parallel python task is executed
            execute_function (Callable, optional): The function to execute tasks. Defaults to execute_in_subprocess.
            terminate_function (Callable, optional): The function to terminate the tasks.
            pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
            backend (str, optional): name of the backend used to spawn tasks.
            disable_dependencies (boolean): Disable resolving future objects during the submission.
        """
        super().__init__(max_cores=None)
        default_resource_dict = {
            "cores": 1,
            "cwd": None,
        }
        if resource_dict is None:
            resource_dict = {}
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        if execute_function == execute_in_subprocess and terminate_function is None:
            terminate_function = terminate_subprocess
        cache_directory_path = os.path.abspath(cache_directory)
        os.makedirs(cache_directory_path, exist_ok=True)
        self._process_kwargs = {
            "future_queue": self._future_queue,
            "execute_function": execute_function,
            "cache_directory": cache_directory_path,
            "resource_dict": resource_dict,
            "terminate_function": terminate_function,
            "pysqa_config_directory": pysqa_config_directory,
            "backend": backend,
            "disable_dependencies": disable_dependencies,
        }
        self._set_process(
            Thread(
                target=execute_tasks_h5,
                kwargs=self._process_kwargs,
            )
        )


def create_file_executor(
    max_workers: Optional[int] = None,
    backend: str = "flux_submission",
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    flux_executor=None,
    flux_executor_pmi_mode: Optional[str] = None,
    flux_executor_nesting: bool = False,
    flux_log_files: bool = False,
    pysqa_config_directory: Optional[str] = None,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
    disable_dependencies: bool = False,
):
    if cache_directory is None:
        cache_directory = "executorlib_cache"
    if block_allocation:
        raise ValueError(
            "The option block_allocation is not available with the pysqa based backend."
        )
    if init_function is not None:
        raise ValueError(
            "The option to specify an init_function is not available with the pysqa based backend."
        )
    check_flux_executor_pmi_mode(flux_executor_pmi_mode=flux_executor_pmi_mode)
    check_max_workers_and_cores(max_cores=max_cores, max_workers=max_workers)
    check_hostname_localhost(hostname_localhost=hostname_localhost)
    check_executor(executor=flux_executor)
    check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
    check_flux_log_files(flux_log_files=flux_log_files)
    return FileTaskScheduler(
        cache_directory=cache_directory,
        resource_dict=resource_dict,
        pysqa_config_directory=pysqa_config_directory,
        backend=backend.split("_submission")[0],
        disable_dependencies=disable_dependencies,
    )
