from concurrent.futures import Future
from queue import Queue, Empty
from time import sleep

from pympipool import Executor
from pympipool.shared.executorbase import ExecutorSteps
from pympipool.shared.thread import RaisingThread


def run_task_with_dependencies(future_queue: Queue, executor_queue: Queue, sleep_interval: float = 0.1):
    wait_lst = []
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except Empty:
            task_dict = None
        if task_dict is not None and "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            executor_queue.put(task_dict)
            future_queue.task_done()
            future_queue.join()
            break
        elif task_dict is not None and "fn" in task_dict.keys() and "future" in task_dict.keys():
            future_lst = (
                    [arg for arg in task_dict["args"] if isinstance(arg, Future)]
                    + [value for value in task_dict["kwargs"] if isinstance(value, Future)]
            )
            result_lst = [future for future in future_lst if future.done()]
            if len(future_lst) == 0 or len(future_lst) == len(result_lst):
                task_dict["args"] = [
                    arg if not isinstance(arg, Future) else arg.result()
                    for arg in task_dict["args"]
                ]
                task_dict["kwargs"] = {
                    key: value if not isinstance(value, Future) else value.result()
                    for key, value in task_dict["kwargs"].items()
                }
                executor_queue.put(task_dict)
            else:
                task_dict["future_lst"] = future_lst
                wait_lst.append(task_dict)
            future_queue.task_done()
        elif len(wait_lst) > 0:
            wait_tmp_lst = []
            for task_wait_dict in wait_lst:
                if all([future.done() for future in task_dict["future_lst"]]):
                    del task_dict["future_lst"]
                    executor_queue.put(task_dict)
                else:
                    wait_tmp_lst.append(task_wait_dict)
            wait_lst = wait_tmp_lst
        else:
            sleep(sleep_interval)


class ExecutorWithDependencies(ExecutorSteps):
    def __init__(self, *args, sleep_interval: float = 0.1, **kwargs):
        super().__init__()
        self._executor_internal = Executor(*args, **kwargs)
        self._set_process(
            RaisingThread(
                target=run_task_with_dependencies,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "executor_queue": self._executor_internal._future_queue,
                    "sleep_interval": sleep_interval,
                },
            )
        )
