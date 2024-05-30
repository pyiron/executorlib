from concurrent.futures import Future
import hashlib
import os
import queue
import re
import subprocess
from typing import Tuple

import cloudpickle

from pympipool.cache.hdf import dump, load, get_output


class FutureItem:
    def __init__(self, file_name: str):
        self._file_name = file_name

    def result(self):
        exec_flag, result = get_output(file_name=self._file_name)
        if exec_flag:
            return result
        else:
            return self.result()

    def done(self):
        return get_output(file_name=self._file_name)[0]


def backend_load_file(file_name: str) -> dict:
    apply_dict = load(file_name=file_name)
    apply_dict["args"] = [
        arg if not isinstance(arg, FutureItem) else arg.result()
        for arg in apply_dict["args"]
    ]
    apply_dict["kwargs"] = {
        key: arg if not isinstance(arg, FutureItem) else arg.result()
        for key, arg in apply_dict["kwargs"].items()
    }
    return apply_dict


def backend_write_file(file_name: str, output):
    file_name_out = os.path.splitext(file_name)[0]
    os.rename(file_name, file_name_out + ".h5ready")
    dump(file_name=file_name_out + ".h5ready", data_dict={"output": output})
    os.rename(file_name_out + ".h5ready", file_name_out + ".h5out")


def execute_in_subprocess(
    command: list, task_dependent_lst: list = []
) -> subprocess.Popen:
    while len(task_dependent_lst) > 0:
        task_dependent_lst = [
            task for task in task_dependent_lst if task.poll() is None
        ]
    return subprocess.Popen(command, universal_newlines=True)


def execute_tasks_h5(
    future_queue: queue.Queue, cache_directory: str, execute_function: callable
):
    memory_dict, process_dict, file_name_dict = {}, {}, {}
    while True:
        task_dict = None
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            pass
        if (
            task_dict is not None
            and "shutdown" in task_dict.keys()
            and task_dict["shutdown"]
        ):
            future_queue.task_done()
            future_queue.join()
            break
        elif task_dict is not None:
            task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
                task_dict=task_dict,
                memory_dict=memory_dict,
                file_name_dict=file_name_dict,
            )
            task_key, data_dict = _serialize_funct_h5(
                task_dict["fn"], *task_args, **task_kwargs
            )
            if task_key not in memory_dict.keys():
                if task_key + ".h5out" not in os.listdir(cache_directory):
                    file_name = os.path.join(cache_directory, task_key + ".h5in")
                    dump(file_name=file_name, data_dict=data_dict)
                    process_dict[task_key] = execute_function(
                        command=_get_execute_command(file_name=file_name),
                        task_dependent_lst=[
                            process_dict[k] for k in future_wait_key_lst
                        ],
                    )
                file_name_dict[task_key] = os.path.join(
                    cache_directory, task_key + ".h5out"
                )
                memory_dict[task_key] = task_dict["future"]
            future_queue.task_done()
        else:
            memory_dict = {
                key: _check_task_output(
                    task_key=key, future_obj=value, cache_directory=cache_directory
                )
                for key, value in memory_dict.items()
                if not value.done()
            }


def execute_task_in_file(file_name: str):
    """
    Execute the task stored in a given HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path
    """
    apply_dict = backend_load_file(file_name=file_name)
    result = apply_dict["fn"].__call__(*apply_dict["args"], **apply_dict["kwargs"])
    backend_write_file(
        file_name=file_name,
        output=result,
    )


def _get_execute_command(file_name: str) -> list:
    return ["python", "-m", "pympipool.backend.serial_cache", file_name]


def _get_hash(binary: bytes):
    # Remove specification of jupyter kernel from hash to be deterministic
    binary_no_ipykernel = re.sub(b"(?<=/ipykernel_)(.*)(?=/)", b"", binary)
    return str(hashlib.md5(binary_no_ipykernel).hexdigest())


def _serialize_funct_h5(fn: callable, *args, **kwargs):
    binary_all = cloudpickle.dumps({"fn": fn, "args": args, "kwargs": kwargs})
    task_key = fn.__name__ + _get_hash(binary=binary_all)
    data = {"fn": fn, "args": args, "kwargs": kwargs}
    return task_key, data


def _check_task_output(
    task_key: str, future_obj: Future, cache_directory: str
) -> Future:
    file_name = os.path.join(cache_directory, task_key + ".h5out")
    if not os.path.exists(file_name):
        return future_obj
    exec_flag, result = get_output(file_name=file_name)
    if exec_flag:
        future_obj.set_result(result)
    return future_obj


def _convert_args_and_kwargs(
    task_dict: dict, memory_dict: dict, file_name_dict: dict
) -> Tuple:
    task_args = []
    task_kwargs = {}
    future_wait_key_lst = []
    for arg in task_dict["args"]:
        if isinstance(arg, Future):
            match_found = False
            for k, v in memory_dict.items():
                if arg == v:
                    task_args.append(FutureItem(file_name=file_name_dict[k]))
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_args.append(arg.result())
        else:
            task_args.append(arg)
    for key, arg in task_dict["kwargs"].items():
        if isinstance(arg, Future):
            match_found = False
            for k, v in memory_dict.items():
                if arg == v:
                    task_kwargs[key] = FutureItem(file_name=file_name_dict[k])
                    future_wait_key_lst.append(k)
                    match_found = True
                    break
            if not match_found:
                task_kwargs[key] = arg.result()
        else:
            task_kwargs[key] = arg
    return task_args, task_kwargs, future_wait_key_lst
