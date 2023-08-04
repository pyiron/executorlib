from time import sleep
import queue

from pympipool.shared.executorbase import (
    ExecutorBase,
    get_future_done,
    execute_task_dict,
)
from pympipool.shared.thread import RaisingThread
from pympipool.mpi.mpitask import MPISingleTaskExecutor


class MPIExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "oversubscribe": oversubscribe,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
            },
        )
        self._process.start()


def executor_broker(
    future_queue,
    max_workers,
    cores_per_worker=1,
    oversubscribe=False,
    init_function=None,
    cwd=None,
    sleep_interval=0.1,
):
    meta_future_lst = get_executor_list(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        oversubscribe=oversubscribe,
        init_function=init_function,
        cwd=cwd,
    )
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            sleep(sleep_interval)
        else:
            if execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def get_executor_list(
    max_workers,
    cores_per_worker=1,
    oversubscribe=False,
    init_function=None,
    cwd=None,
):
    return {
        get_future_done(): MPISingleTaskExecutor(
            cores=cores_per_worker,
            oversubscribe=oversubscribe,
            init_function=init_function,
            cwd=cwd,
        )
        for _ in range(max_workers)
    }
