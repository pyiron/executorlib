from concurrent.futures import Future
import os
import queue
import sys
from time import sleep
from typing import Optional, List

from executorlib_core.communication import interface_bootup
from executorlib_core.thread import RaisingThread
from executorlib_core.inputcheck import (
    check_resource_dict,
    check_resource_dict_is_empty,
)
from executorlib_core.base import ExecutorBase

from pympipool.shared.interface import BaseInterface


class ExecutorSteps(ExecutorBase):
    def submit(self, fn: callable, *args, resource_dict: dict = {}, **kwargs):
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): resource dictionary, which defines the resources used for the execution of the
                                  function. Example resource dictionary: {
                                      cores: 1,
                                      threads_per_core: 1,
                                      gpus_per_worker: 0,
                                      oversubscribe: False,
                                      cwd: None,
                                      executor: None,
                                      hostname_localhost: False,
                                  }

        Returns:
            A Future representing the given call.
        """
        check_resource_dict(function=fn)
        f = Future()
        self._future_queue.put(
            {
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "future": f,
                "resource_dict": resource_dict,
            }
        )
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
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
        if self._process is not None:
            self._future_queue.put({"shutdown": True, "wait": wait})
            if wait:
                self._process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None


def execute_parallel_tasks(
    future_queue: queue.Queue,
    cores: int,
    interface_class: BaseInterface,
    hostname_localhost: bool = False,
    init_function: Optional[callable] = None,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       interface_class (BaseInterface): Interface to start process on selected compute resources
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
       init_function (callable): optional function to preset arguments for functions which are submitted later
    """
    interface = interface_bootup(
        command_lst=_get_backend_path(cores=cores),
        connections=interface_class(cores=cores, **kwargs),
        hostname_localhost=hostname_localhost,
    )
    if init_function is not None:
        interface.send_dict(
            input_dict={"init": True, "fn": init_function, "args": (), "kwargs": {}}
        )
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
                except Exception as thread_exception:
                    interface.shutdown(wait=True)
                    future_queue.task_done()
                    f.set_exception(exception=thread_exception)
                    raise thread_exception
                else:
                    future_queue.task_done()


def execute_separate_tasks(
    future_queue: queue.Queue,
    interface_class: BaseInterface,
    max_cores: int,
    hostname_localhost: bool = False,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       interface_class (BaseInterface): Interface to start process on selected compute resources
       max_cores (int): defines the number cores which can be used in parallel
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
    """
    active_task_dict = {}
    process_lst, qtask_lst = [], []
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            if task_dict["wait"]:
                _ = [process.join() for process in process_lst]
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            qtask = queue.Queue()
            process, active_task_dict = _submit_function_to_separate_process(
                task_dict=task_dict,
                qtask=qtask,
                active_task_dict=active_task_dict,
                interface_class=interface_class,
                executor_kwargs=kwargs,
                max_cores=max_cores,
                hostname_localhost=hostname_localhost,
            )
            qtask_lst.append(qtask)
            process_lst.append(process)
            future_queue.task_done()


def execute_tasks_with_dependencies(
    future_queue: queue.Queue,
    executor_queue: queue.Queue,
    executor: ExecutorBase,
    refresh_rate: float = 0.01,
):
    """
    Resolve the dependencies of multiple tasks, by analysing which task requires concurrent.future.Futures objects from
    other tasks.

    Args:
        future_queue (Queue): Queue for receiving new tasks.
        executor_queue (Queue): Queue for the internal executor.
        executor (ExecutorBase): Executor to execute the tasks with after the dependencies are resolved.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
    """
    wait_lst = []
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            task_dict = None
        if (  # shutdown the executor
            task_dict is not None
            and "shutdown" in task_dict.keys()
            and task_dict["shutdown"]
        ):
            executor.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        elif (  # handle function submitted to the executor
            task_dict is not None
            and "fn" in task_dict.keys()
            and "future" in task_dict.keys()
        ):
            future_lst, ready_flag = _get_future_objects_from_input(task_dict=task_dict)
            if len(future_lst) == 0 or ready_flag:
                # No future objects are used in the input or all future objects are already done
                task_dict["args"], task_dict["kwargs"] = _update_futures_in_input(
                    args=task_dict["args"], kwargs=task_dict["kwargs"]
                )
                executor_queue.put(task_dict)
            else:  # Otherwise add the function to the wait list
                task_dict["future_lst"] = future_lst
                wait_lst.append(task_dict)
            future_queue.task_done()
        elif len(wait_lst) > 0:
            # Check functions in the wait list and execute them if all future objects are now ready
            wait_lst = _submit_waiting_task(
                wait_lst=wait_lst, executor_queue=executor_queue
            )
        else:
            # If there is nothing else to do, sleep for a moment
            sleep(refresh_rate)


def _get_backend_path(cores: int):
    """
    Get command to call backend as a list of two strings

    Args:
        cores (int): Number of cores used to execute the task, if it is greater than one use mpiexec.py else serial.py

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1:
        command_lst += [_get_command_path(executable="mpiexec.py")]
    else:
        command_lst += [_get_command_path(executable="serial.py")]
    return command_lst


def _get_command_path(executable: str):
    """
    Get path of the backend executable script

    Args:
        executable (str): Name of the backend executable script, either mpiexec.py or serial.py

    Returns:
        str: absolute path to the executable script
    """
    return os.path.abspath(os.path.join(__file__, "..", "..", "backend", executable))


def _wait_for_free_slots(active_task_dict: dict, cores_requested: int, max_cores: int):
    """
    Wait for available computing resources to become available.

    Args:
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        cores_requested (int): Number of cores required for executing the next task
        max_cores (int): Maximum number cores which can be used

    Returns:
        dict: Dictionary containing the future objects and the number of cores they require
    """
    while sum(active_task_dict.values()) + cores_requested > max_cores:
        active_task_dict = {k: v for k, v in active_task_dict.items() if not k.done()}
    return active_task_dict


def _submit_waiting_task(wait_lst: List[dict], executor_queue: queue.Queue):
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
            task_wait_dict["args"], task_wait_dict["kwargs"] = _update_futures_in_input(
                args=task_wait_dict["args"], kwargs=task_wait_dict["kwargs"]
            )
            executor_queue.put(task_wait_dict)
        else:
            wait_tmp_lst.append(task_wait_dict)
    return wait_tmp_lst


def _update_futures_in_input(args: tuple, kwargs: dict):
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


def _get_future_objects_from_input(task_dict: dict):
    """
    Check the input parameters if they contain future objects and which of these future objects are executed

    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}

    Returns:
        list, boolean: list of future objects and boolean flag if all future objects are already done
    """
    future_lst = [arg for arg in task_dict["args"] if isinstance(arg, Future)] + [
        value for value in task_dict["kwargs"] if isinstance(value, Future)
    ]
    boolean_flag = len([future for future in future_lst if future.done()]) == len(
        future_lst
    )
    return future_lst, boolean_flag


def _submit_function_to_separate_process(
    task_dict: dict,
    active_task_dict: dict,
    qtask: queue.Queue,
    interface_class: BaseInterface,
    executor_kwargs: dict,
    max_cores: int,
    hostname_localhost: bool = False,
):
    """
    Submit function to be executed in separate Python process
    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        qtask (queue.Queue): Queue to communicate with the thread linked to the process executing the python function
        interface_class (BaseInterface): Interface to start process on selected compute resources
        executor_kwargs (dict): keyword parameters used to initialize the Executor
        max_cores (int): defines the number cores which can be used in parallel
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
    Returns:
        RaisingThread, dict: thread for communicating with the python process which is executing the function and
                             dictionary containing the future objects and the number of cores they require
    """
    resource_dict = task_dict.pop("resource_dict")
    qtask.put(task_dict)
    qtask.put({"shutdown": True, "wait": True})
    if "cores" not in resource_dict.keys() or (
        resource_dict["cores"] == 1 and executor_kwargs["cores"] >= 1
    ):
        resource_dict["cores"] = executor_kwargs["cores"]
    active_task_dict = _wait_for_free_slots(
        active_task_dict=active_task_dict,
        cores_requested=resource_dict["cores"],
        max_cores=max_cores,
    )
    active_task_dict[task_dict["future"]] = resource_dict["cores"]
    task_kwargs = executor_kwargs.copy()
    task_kwargs.update(resource_dict)
    task_kwargs.update(
        {
            "future_queue": qtask,
            "interface_class": interface_class,
            "hostname_localhost": hostname_localhost,
            "init_function": None,
        }
    )
    process = RaisingThread(
        target=execute_parallel_tasks,
        kwargs=task_kwargs,
    )
    process.start()
    return process, active_task_dict
