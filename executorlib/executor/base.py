import queue
from abc import ABC
from concurrent.futures import (
    Executor as FutureExecutor,
)
from concurrent.futures import (
    Future,
)
from typing import Callable, Optional

from executorlib.task_scheduler.base import TaskSchedulerBase


class BaseExecutor(FutureExecutor, ABC):
    """
    Interface class for the executor.

    Args:
        executor (TaskSchedulerBase): internal executor
    """

    def __init__(self, executor: TaskSchedulerBase):
        self._task_scheduler = executor

    @property
    def max_workers(self) -> Optional[int]:
        return self._task_scheduler.max_workers

    @max_workers.setter
    def max_workers(self, max_workers: int):
        self._task_scheduler.max_workers = max_workers

    @property
    def info(self) -> Optional[dict]:
        """
        Get the information about the executor.

        Returns:
            Optional[dict]: Information about the executor.
        """
        return self._task_scheduler.info

    @property
    def future_queue(self) -> Optional[queue.Queue]:
        """
        Get the future queue.

        Returns:
            queue.Queue: The future queue.
        """
        return self._task_scheduler.future_queue

    def submit(  # type: ignore
        self,
        fn: Callable,
        /,
        *args,
        resource_dict: Optional[dict] = None,
        **kwargs,
    ) -> Future:
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): resource dictionary, which defines the resources used for the execution of the
                                  function. Example resource dictionary: {
                                      cores: 1,
                                      threads_per_core: 1,
                                      gpus_per_worker: 0,
                                      oversubscribe: False,
                                      cwd: None,
                                      executor: None,
                                      hostname_localhost: False,
                                  }

        Returns:
            Future: A Future representing the given call.
        """
        return self._task_scheduler.submit(
            *([fn] + list(args)), resource_dict=resource_dict, **kwargs
        )

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """
        Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait (bool): If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                parallel_executors have been reclaimed.
            cancel_futures (bool): If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        self._task_scheduler.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __len__(self) -> int:
        """
        Get the length of the executor.

        Returns:
            int: The length of the executor.
        """
        return len(self._task_scheduler)

    def __bool__(self):
        """
        Overwrite length to always return True

        Returns:
            bool: Always return True
        """
        return True

    def __exit__(self, *args, **kwargs) -> None:
        """
        Exit method called when exiting the context manager.
        """
        self._task_scheduler.__exit__(*args, **kwargs)
