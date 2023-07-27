from concurrent.futures import Executor as FutureExecutor, Future
import queue

from pympipool.shared.thread import RaisingThread
from pympipool.shared.broker import executor_broker
from pympipool.shared.taskexecutor import cancel_items_in_queue


class HPCExecutor(FutureExecutor):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        gpus_per_worker=0,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_slurm_backend=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        self._future_queue = queue.Queue()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "gpus_per_worker": gpus_per_worker,
                "oversubscribe": oversubscribe,
                "enable_flux_backend": enable_flux_backend,
                "enable_slurm_backend": enable_slurm_backend,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "queue_adapter": queue_adapter,
                "queue_adapter_kwargs": queue_adapter_kwargs,
            },
        )
        self._process.start()

    def submit(self, fn, *args, **kwargs):
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        f = Future()
        self._future_queue.put({"fn": fn, "args": args, "kwargs": kwargs, "future": f})
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                parallel_executors have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        if cancel_futures:
            cancel_items_in_queue(que=self._future_queue)
        self._future_queue.put({"shutdown": True, "wait": wait})
        self._process.join()
