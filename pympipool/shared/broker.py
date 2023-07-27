from concurrent.futures import as_completed, Future
import queue
from time import sleep

from pympipool.interfaces.taskexecutor import Executor


class MetaExecutorFuture(object):
    def __init__(self, future, executor):
        self.future = future
        self.executor = executor

    @property
    def _condition(self):
        return self.future._condition

    @property
    def _state(self):
        return self.future._state

    @property
    def _waiters(self):
        return self.future._waiters

    def done(self):
        return self.future.done()

    def submit(self, task_dict):
        self.future = task_dict["future"]
        self.executor._future_queue.put(task_dict)


def executor_broker(
    future_queue,
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
    meta_future_lst = _get_executor_list(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        gpus_per_worker=gpus_per_worker,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_slurm_backend=enable_slurm_backend,
        init_function=init_function,
        cwd=cwd,
        queue_adapter=queue_adapter,
        queue_adapter_kwargs=queue_adapter_kwargs,
    )
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            sleep(sleep_interval)
        else:
            if _execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def _execute_task_dict(task_dict, meta_future_lst):
    if "fn" in task_dict.keys():
        meta_future = next(as_completed(meta_future_lst))
        meta_future.submit(task_dict=task_dict)
        return True
    elif "shutdown" in task_dict.keys() and task_dict["shutdown"]:
        for meta in meta_future_lst:
            meta.executor.shutdown(wait=task_dict["wait"])
        return False
    else:
        raise ValueError("Unrecognized Task in task_dict: ", task_dict)


def _get_executor_list(
    max_workers,
    cores_per_worker=1,
    gpus_per_worker=0,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    init_function=None,
    cwd=None,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    return [
        MetaExecutorFuture(
            future=_get_future_done(),
            executor=Executor(
                cores=cores_per_worker,
                gpus_per_task=int(gpus_per_worker / cores_per_worker),
                oversubscribe=oversubscribe,
                enable_flux_backend=enable_flux_backend,
                enable_slurm_backend=enable_slurm_backend,
                init_function=init_function,
                cwd=cwd,
                queue_adapter=queue_adapter,
                queue_adapter_kwargs=queue_adapter_kwargs,
            ),
        )
        for _ in range(max_workers)
    ]


def _get_future_done():
    f = Future()
    f.set_result(True)
    return f
