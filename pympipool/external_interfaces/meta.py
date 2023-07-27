from concurrent.futures import as_completed, Executor as FutureExecutor, Future
import queue
from threading import Thread
from time import sleep

from pympipool.external_interfaces.executor import Executor
from pympipool.shared_functions.external_interfaces import cancel_items_in_queue



class MetaExecutorFuture(object):
    def __init__(self, future, executor):
        self._future = future
        self._executor = executor

    @property
    def _condition(self):
        return self._future._condition

    @property
    def _state(self):
        return self._future._state

    @property
    def _waiters(self):
        return self._future._waiters

    def done(self):
        return self._future.done()

    def submit(self, task_dict):
        self._future = task_dict["future"]
        self._executor._future_queue.put(task_dict)


class MetaExecutor(FutureExecutor):
    def __init__(self, max_workers, cores_per_worker=1, sleep_interval=0.1):
        self._future_queue = queue.Queue()
        self._process = Thread(
            target=background,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "sleep_interval": sleep_interval
            }
        )

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


def get_future_done():
    f = Future()
    f.set_result(True)
    return f


def background(future_queue, max_workers, cores_per_worker=1, sleep_interval=0.1):
    meta_future_lst = [
        MetaExecutorFuture(future=get_future_done(), executor=Executor(cores=cores_per_worker))
        for i in range(max_workers)
    ]
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            sleep(sleep_interval)
        else:
            if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
                for meta in meta_future_lst:
                    meta._executor.shutdown(wait=task_dict["wait"])
                break
            else:
                meta_future = next(as_completed(meta_future_lst))
                meta_future.submit(task_dict=task_dict)
