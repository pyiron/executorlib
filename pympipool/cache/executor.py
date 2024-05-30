import os

from pympipool.shared.executorbase import ExecutorBase
from pympipool.shared.thread import RaisingThread
from pympipool.cache.shared import execute_in_subprocess, execute_tasks_h5


class FileExecutor(ExecutorBase):
    def __init__(self, cache_directory="cache", execute_function=execute_in_subprocess):
        super().__init__()
        cache_directory_path = os.path.abspath(cache_directory)
        os.makedirs(cache_directory_path, exist_ok=True)
        self._set_process(
            RaisingThread(
                target=execute_tasks_h5,
                kwargs={
                    "future_queue": self._future_queue,
                    "execute_function": execute_function,
                    "cache_directory": cache_directory_path,
                },
            )
        )
