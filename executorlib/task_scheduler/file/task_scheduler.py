from threading import Thread
from typing import Callable, Optional

from executorlib.standalone.inputcheck import (
    check_executor,
    check_flux_log_files,
    check_hostname_localhost,
    check_max_workers_and_cores,
    check_nested_flux_executor,
    check_pmi_mode,
)
from executorlib.task_scheduler.base import TaskSchedulerBase
from executorlib.task_scheduler.file.shared import execute_tasks_h5
from executorlib.task_scheduler.file.subprocess_spawner import (
    execute_in_subprocess,
    terminate_subprocess,
)

try:
    from executorlib.standalone.scheduler import terminate_with_pysqa
    from executorlib.task_scheduler.file.queue_spawner import execute_with_pysqa
except ImportError:
    # If pysqa is not available fall back to executing tasks in a subprocess
    execute_with_pysqa = execute_in_subprocess  # type: ignore
    terminate_with_pysqa = None  # type: ignore


class FileTaskScheduler(TaskSchedulerBase):
    def __init__(
        self,
        resource_dict: Optional[dict] = None,
        execute_function: Callable = execute_with_pysqa,
        terminate_function: Optional[Callable] = None,
        pysqa_config_directory: Optional[str] = None,
        backend: Optional[str] = None,
        disable_dependencies: bool = False,
        pmi_mode: Optional[str] = None,
    ):
        """
        Initialize the FileExecutor.

        Args:
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - cache_directory (str): The directory to store cache files.
            execute_function (Callable, optional): The function to execute tasks. Defaults to execute_in_subprocess.
            terminate_function (Callable, optional): The function to terminate the tasks.
            pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
            backend (str, optional): name of the backend used to spawn tasks.
            disable_dependencies (boolean): Disable resolving future objects during the submission.
            pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
        """
        super().__init__(max_cores=None)
        default_resource_dict = {
            "cores": 1,
            "cwd": None,
            "cache_directory": "executorlib_cache",
        }
        if resource_dict is None:
            resource_dict = {}
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        self._process_kwargs = {
            "resource_dict": resource_dict,
            "future_queue": self._future_queue,
            "execute_function": execute_function,
            "terminate_function": terminate_function,
            "pysqa_config_directory": pysqa_config_directory,
            "backend": backend,
            "disable_dependencies": disable_dependencies,
            "pmi_mode": pmi_mode,
        }
        self._set_process(
            Thread(
                target=execute_tasks_h5,
                kwargs=self._process_kwargs,
            )
        )


def create_file_executor(
    resource_dict: dict,
    max_workers: Optional[int] = None,
    backend: Optional[str] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    pmi_mode: Optional[str] = None,
    flux_executor=None,
    flux_executor_nesting: bool = False,
    flux_log_files: bool = False,
    pysqa_config_directory: Optional[str] = None,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
    disable_dependencies: bool = False,
    execute_function: Callable = execute_with_pysqa,
):
    if block_allocation:
        raise ValueError(
            "The option block_allocation is not available with the pysqa based backend."
        )
    if init_function is not None:
        raise ValueError(
            "The option to specify an init_function is not available with the pysqa based backend."
        )
    if cache_directory is not None:
        resource_dict["cache_directory"] = cache_directory
    if backend is None:
        check_pmi_mode(pmi_mode=pmi_mode)
    check_max_workers_and_cores(max_cores=max_cores, max_workers=max_workers)
    check_hostname_localhost(hostname_localhost=hostname_localhost)
    check_executor(executor=flux_executor)
    check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
    check_flux_log_files(flux_log_files=flux_log_files)
    if execute_function != execute_in_subprocess:
        terminate_function = terminate_with_pysqa  # type: ignore
    else:
        terminate_function = terminate_subprocess  # type: ignore
    return FileTaskScheduler(
        resource_dict=resource_dict,
        pysqa_config_directory=pysqa_config_directory,
        backend=backend,
        disable_dependencies=disable_dependencies,
        execute_function=execute_function,
        terminate_function=terminate_function,
        pmi_mode=pmi_mode,
    )
