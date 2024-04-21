from concurrent.futures import Future
from queue import Queue, Empty
from time import sleep
from typing import List

from pympipool.scheduler import create_executor
from pympipool.shared.executorbase import ExecutorSteps, ExecutorBase
from pympipool.shared.thread import RaisingThread


def run_task_with_dependencies(
    future_queue: Queue, executor: ExecutorBase, sleep_interval: float = 0.1
):
    wait_lst = []
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except Empty:
            task_dict = None
        if (
            task_dict is not None
            and "shutdown" in task_dict.keys()
            and task_dict["shutdown"]
        ):
            executor.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        elif (
            task_dict is not None
            and "fn" in task_dict.keys()
            and "future" in task_dict.keys()
        ):
            future_lst = [arg for arg in task_dict["args"] if isinstance(arg, Future)] + [
                value for value in task_dict["kwargs"] if isinstance(value, Future)
            ]
            result_lst = [future for future in future_lst if future.done()]
            if len(future_lst) == 0 or len(future_lst) == len(result_lst):
                task_dict["args"], task_dict["kwargs"] = update_futures_in_input(
                    args=task_dict["args"], kwargs=task_dict["kwargs"]
                )
                executor._future_queue.put(task_dict)
            else:
                task_dict["future_lst"] = future_lst
                wait_lst.append(task_dict)
            future_queue.task_done()
        elif len(wait_lst) > 0:
            wait_lst = submit_waiting_task(
                wait_lst=wait_lst, executor_queue=executor._future_queue
            )
        else:
            sleep(sleep_interval)


def submit_waiting_task(wait_lst: List[dict], executor_queue: Queue):
    """
    Submit the waiting tasks, which future inputs have been completed, to the executor

    Args:
        wait_lst (list): List of waiting tasks
        executor_queue (Queue): Queue of the internal executor

    Returns:
        list: list tasks which future inputs have not been completed
    """
    wait_tmp_lst = []
    for task_wait_dict in wait_lst:
        if all([future.done() for future in task_wait_dict["future_lst"]]):
            del task_wait_dict["future_lst"]
            task_wait_dict["args"], task_wait_dict["kwargs"] = update_futures_in_input(
                args=task_wait_dict["args"], kwargs=task_wait_dict["kwargs"]
            )
            executor_queue.put(task_wait_dict)
        else:
            wait_tmp_lst.append(task_wait_dict)
    return wait_tmp_lst


def update_futures_in_input(args: tuple, kwargs: dict):
    """
    Evaluate future objects in the arguments and keyword arguments by calling future.result()

    Args:
        args (tuple): function arguments
        kwargs (dict): function keyword arguments

    Returns:
        tuple, dict: arguments and keyword arguments with each future object in them being evaluated
    """
    args = [arg if not isinstance(arg, Future) else arg.result() for arg in args]
    kwargs = {
        key: value if not isinstance(value, Future) else value.result()
        for key, value in kwargs.items()
    }
    return args, kwargs


class ExecutorWithDependencies(ExecutorSteps):
    def __init__(self, *args, sleep_interval: float = 0.1, **kwargs):
        super().__init__()
        self._set_process(
            RaisingThread(
                target=run_task_with_dependencies,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "executor": create_executor(*args, **kwargs),
                    "sleep_interval": sleep_interval,
                },
            )
        )