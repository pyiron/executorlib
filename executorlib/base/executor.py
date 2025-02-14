import contextlib
import queue
from concurrent.futures import (
    Executor as FutureExecutor,
)
from concurrent.futures import (
    Future,
)
from threading import Thread
from typing import Callable, Optional, Union

from executorlib.standalone.inputcheck import check_resource_dict
from executorlib.standalone.queue import cancel_items_in_queue
from executorlib.standalone.serialize import cloudpickle_register


class ExecutorBase(FutureExecutor):
    """
    Base class for the executor.

    Args:
        max_cores (int): defines the number cores which can be used in parallel
    """

    def __init__(self, max_cores: Optional[int] = None):
        """
        Initialize the ExecutorBase class.
        """
        cloudpickle_register(ind=3)
        self._max_cores = max_cores
        self._future_queue: Optional[queue.Queue] = queue.Queue()
        self._process: Optional[Union[Thread, list[Thread]]] = None

    @property
    def info(self) -> Optional[dict]:
        """
        Get the information about the executor.

        Returns:
            Optional[dict]: Information about the executor.
        """
        if self._process is not None and isinstance(self._process, list):
            meta_data_dict = self._process[0]._kwargs.copy()
            if "future_queue" in meta_data_dict:
                del meta_data_dict["future_queue"]
            meta_data_dict["max_workers"] = len(self._process)
            return meta_data_dict
        elif self._process is not None:
            meta_data_dict = self._process._kwargs.copy()
            if "future_queue" in meta_data_dict:
                del meta_data_dict["future_queue"]
            return meta_data_dict
        else:
            return None

    @property
    def future_queue(self) -> Optional[queue.Queue]:
        """
        Get the future queue.

        Returns:
            queue.Queue: The future queue.
        """
        return self._future_queue

    def submit(  # type: ignore
        self,
        fn: Callable,
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
        if resource_dict is None:
            resource_dict = {}
        cores = resource_dict.get("cores")
        if (
            cores is not None
            and self._max_cores is not None
            and cores > self._max_cores
        ):
            raise ValueError(
                "The specified number of cores is larger than the available number of cores."
            )
        check_resource_dict(function=fn)
        f: Future = Future()
        if self._future_queue is not None:
            self._future_queue.put(
                {
                    "fn": fn,
                    "args": args,
                    "kwargs": kwargs,
                    "future": f,
                    "resource_dict": resource_dict,
                }
            )
        return f

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
        if cancel_futures and self._future_queue is not None:
            cancel_items_in_queue(que=self._future_queue)
        if self._process is not None and self._future_queue is not None:
            self._future_queue.put({"shutdown": True, "wait": wait})
            if wait and isinstance(self._process, Thread):
                self._process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: Thread):
        """
        Set the process for the executor.

        Args:
            process (RaisingThread): The process for the executor.
        """
        self._process = process
        self._process.start()

    def __len__(self) -> int:
        """
        Get the length of the executor.

        Returns:
            int: The length of the executor.
        """
        queue_size = 0
        if self._future_queue is not None:
            queue_size = self._future_queue.qsize()
        return queue_size

    def __del__(self):
        """
        Clean-up the resources associated with the Executor.
        """
        with contextlib.suppress(AttributeError, RuntimeError):
            self.shutdown(wait=False)
