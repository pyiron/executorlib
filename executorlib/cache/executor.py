import os
from typing import Optional

from executorlib.base.executor import ExecutorBase
from executorlib.cache.shared import (
    execute_in_subprocess,
    execute_tasks_h5,
    terminate_subprocess,
)
from executorlib.standalone.thread import RaisingThread


class FileExecutor(ExecutorBase):
    def __init__(
        self,
        cache_directory: str = "cache",
        execute_function: callable = execute_in_subprocess,
        cores_per_worker: int = 1,
        terminate_function: Optional[callable] = None,
    ):
        """
        Initialize the FileExecutor.

        Args:
            cache_directory (str, optional): The directory to store cache files. Defaults to "cache".
            execute_function (callable, optional): The function to execute tasks. Defaults to execute_in_subprocess.
            cores_per_worker (int, optional): The number of CPU cores per worker. Defaults to 1.
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
                    "terminate_function": terminate_function,
                },
            )
        )
