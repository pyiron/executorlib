import os
from typing import Optional

from executorlib.base.executor import ExecutorBase
from executorlib.cache.shared import execute_tasks_h5
from executorlib.standalone.cache.spawner import (
    execute_in_subprocess,
    terminate_subprocess,
)
from executorlib.standalone.thread import RaisingThread


class FileExecutor(ExecutorBase):
    def __init__(
        self,
        cache_directory: str = "cache",
        cores_per_worker: int = 1,
        cwd: Optional[str] = None,
        execute_function: callable = execute_in_subprocess,
        terminate_function: Optional[callable] = None,
        config_directory: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize the FileExecutor.

        Args:
            cache_directory (str, optional): The directory to store cache files. Defaults to "cache".
            execute_function (callable, optional): The function to execute tasks. Defaults to execute_in_subprocess.
            cores_per_worker (int, optional): The number of CPU cores per worker. Defaults to 1.
            terminate_function (callable, optional): The function to terminate the tasks.
            cwd (str, optional): current working directory where the parallel python task is executed
            config_directory (str, optional): path to the config directory.
            backend (str, optional): name of the backend used to spawn tasks.
        """
        super().__init__()
        if execute_function == execute_in_subprocess and terminate_function is None:
            terminate_function = terminate_subprocess
        cache_directory_path = os.path.abspath(cache_directory)
        os.makedirs(cache_directory_path, exist_ok=True)
        self._set_process(
            RaisingThread(
                target=execute_tasks_h5,
                kwargs={
                    "future_queue": self._future_queue,
                    "execute_function": execute_function,
                    "cache_directory": cache_directory_path,
                    "cores_per_worker": cores_per_worker,
                    "cwd": cwd,
                    "terminate_function": terminate_function,
                    "config_directory": config_directory,
                    "backend": backend,
                },
            )
        )
