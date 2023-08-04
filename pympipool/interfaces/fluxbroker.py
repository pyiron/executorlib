import queue
from time import sleep

from pympipool.shared.executorbase import ExecutorBase
from pympipool.shared.thread import RaisingThread
from pympipool.interfaces.fluxtask import FluxSingleTaskExecutor
from pympipool.shared.executorbase import execute_task_dict, get_future_done


class PyFluxExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        executor=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "threads_per_core": threads_per_core,
                "gpus_per_worker": gpus_per_worker,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "executor": executor,
            },
        )
        self._process.start()


def executor_broker(
    future_queue,
    max_workers,
    cores_per_worker=1,
    threads_per_core=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    sleep_interval=0.1,
    executor=None,
):
    meta_future_lst = _get_executor_list(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        threads_per_core=threads_per_core,
        gpus_per_worker=gpus_per_worker,
        init_function=init_function,
        cwd=cwd,
        executor=executor,
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


def _get_executor_list(
    max_workers,
    cores_per_worker=1,
    threads_per_core=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    executor=None,
):
    return {
        get_future_done(): FluxSingleTaskExecutor(
            cores=cores_per_worker,
            threads_per_core=threads_per_core,
            gpus_per_task=int(gpus_per_worker / cores_per_worker),
            init_function=init_function,
            cwd=cwd,
            executor=executor,
        )
        for _ in range(max_workers)
    }
