import time
import queue

from pympipool.shared.broker import execute_task_dict, get_future_done
from pympipool.legacy.interfaces.taskexecutor import Executor


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
    meta_future_lst = get_executor_list(
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
            time.sleep(sleep_interval)
        else:
            if execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def get_executor_list(
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
