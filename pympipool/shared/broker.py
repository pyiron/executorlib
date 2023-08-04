from concurrent.futures import as_completed, Future
import queue
from time import sleep

from pympipool.interfaces.taskexecutor import Executor


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
            if execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def execute_task_dict(task_dict, meta_future_lst):
    if "fn" in task_dict.keys():
        meta_future = next(as_completed(meta_future_lst.keys()))
        executor = meta_future_lst.pop(meta_future)
        executor.future_queue.put(task_dict)
        meta_future_lst[task_dict["future"]] = executor
        return True
    elif "shutdown" in task_dict.keys() and task_dict["shutdown"]:
        for executor in meta_future_lst.values():
            executor.shutdown(wait=task_dict["wait"])
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
    return {
        get_future_done(): Executor(
            cores=cores_per_worker,
            gpus_per_task=int(gpus_per_worker / cores_per_worker),
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            init_function=init_function,
            cwd=cwd,
            queue_adapter=queue_adapter,
            queue_adapter_kwargs=queue_adapter_kwargs,
        )
        for _ in range(max_workers)
    }


def get_future_done():
    f = Future()
    f.set_result(True)
    return f
