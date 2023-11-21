from concurrent.futures import Future
import subprocess

from pympipool.shared.executorbase import (
    executor_broker, ExecutorBase
)
from pympipool.shared.thread import RaisingThread


def execute_single_task(future_queue):
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            future_queue.task_done()
            future_queue.join()
            break
        elif "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(
                        subprocess.check_output(
                            *task_dict["args"], **task_dict["kwargs"]
                        )
                    )
                except Exception as thread_exception:
                    future_queue.task_done()
                    f.set_exception(exception=thread_exception)
                    raise thread_exception
                else:
                    future_queue.task_done()
        else:
            raise KeyError(task_dict)


class ShellStaticExecutor(ExecutorBase):
    def __init__(self):
        super().__init__()
        self._process = RaisingThread(
            target=execute_single_task,
            kwargs={
                "future_queue": self._future_queue,
            },
        )
        self._process.start()

    def submit(self, *args, **kwargs):
        f = Future()
        self._future_queue.put({"future": f, "args": args, "kwargs": kwargs})
        return f


class ShellExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers=1,
        sleep_interval=0.1,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                # Broker Arguments
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "sleep_interval": sleep_interval,
                "executor_class": ShellStaticExecutor,
            },
        )
        self._process.start()

    def submit(self, *args, **kwargs):
        f = Future()
        self._future_queue.put({"future": f, "args": args, "kwargs": kwargs})
        return f
