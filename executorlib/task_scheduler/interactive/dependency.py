import queue
from concurrent.futures import Future
from threading import Thread
from time import sleep
from typing import Any, Callable, Optional

from executorlib.standalone.batched import batched_futures
from executorlib.standalone.interactive.arguments import (
    check_exception_was_raised,
    get_exception_lst,
    get_future_objects_from_input,
    update_futures_in_input,
)
from executorlib.standalone.plot import (
    generate_nodes_and_edges_for_plotting,
    generate_task_hash_for_plotting,
    plot_dependency_graph_function,
)
from executorlib.task_scheduler.base import TaskSchedulerBase


class DependencyTaskScheduler(TaskSchedulerBase):
    """
    ExecutorWithDependencies is a class that extends ExecutorBase and provides functionality for executing tasks with
    dependencies.

    Args:
        refresh_rate (float, optional): The refresh rate for updating the executor queue. Defaults to 0.01.
        plot_dependency_graph (bool, optional): Whether to generate and plot the dependency graph. Defaults to False.
        plot_dependency_graph_filename (str): Name of the file to store the plotted graph in.

    Attributes:
        _future_hash_dict (Dict[str, Future]): A dictionary mapping task hash to future object.
        _task_hash_dict (Dict[str, Dict]): A dictionary mapping task hash to task dictionary.
        _generate_dependency_graph (bool): Whether to generate the dependency graph.
        _generate_dependency_graph (str): Name of the file to store the plotted graph in.

    """

    def __init__(
        self,
        executor: TaskSchedulerBase,
        max_cores: Optional[int] = None,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        plot_dependency_graph_filename: Optional[str] = None,
    ) -> None:
        super().__init__(max_cores=max_cores)
        self._process_kwargs = {
            "future_queue": self._future_queue,
            "executor_queue": executor._future_queue,
            "executor": executor,
            "refresh_rate": refresh_rate,
        }
        self._set_process(
            Thread(
                target=_execute_tasks_with_dependencies,
                kwargs=self._process_kwargs,
            )
        )
        self._future_hash_dict: dict = {}
        self._task_hash_dict: dict = {}
        self._plot_dependency_graph_filename = plot_dependency_graph_filename
        if plot_dependency_graph_filename is None:
            self._generate_dependency_graph = plot_dependency_graph
        else:
            self._generate_dependency_graph = True

    @property
    def info(self) -> Optional[dict]:
        """
        Get the information about the executor.

        Returns:
            Optional[dict]: Information about the executor.
        """
        if isinstance(self._future_queue, queue.Queue):
            f: Future = Future()
            self._future_queue.queue.insert(
                0, {"internal": True, "task": "get_info", "future": f}
            )
            return f.result()
        else:
            return None

    @property
    def max_workers(self) -> Optional[int]:
        if isinstance(self._future_queue, queue.Queue):
            f: Future = Future()
            self._future_queue.queue.insert(
                0, {"internal": True, "task": "get_max_workers", "future": f}
            )
            return f.result()
        else:
            return None

    @max_workers.setter
    def max_workers(self, max_workers: int):
        if isinstance(self._future_queue, queue.Queue):
            f: Future = Future()
            self._future_queue.queue.insert(
                0,
                {
                    "internal": True,
                    "task": "set_max_workers",
                    "max_workers": max_workers,
                    "future": f,
                },
            )
            if not f.result():
                raise NotImplementedError("The max_workers setter is not implemented.")

    def submit(  # type: ignore
        self,
        fn: Callable[..., Any],
        *args: Any,
        resource_dict: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Future:
        """
        Submits a task to the executor.

        Args:
            fn (Callable): The function to be executed.
            *args: Variable length argument list.
            resource_dict (dict, optional): A dictionary of resources required by the task. Defaults to {}.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Future: A future object representing the result of the task.

        """
        if resource_dict is None:
            resource_dict = {}
        if not self._generate_dependency_graph:
            f = super().submit(fn, *args, resource_dict=resource_dict, **kwargs)
        else:
            f = Future()
            f.set_result(None)
            task_dict = {
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "future": f,
                "resource_dict": resource_dict,
            }
            task_hash = generate_task_hash_for_plotting(
                task_dict=task_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            self._future_hash_dict[task_hash] = f
            self._task_hash_dict[task_hash] = task_dict
        return f

    def batched(
        self,
        iterable: list[Future],
        n: int,
    ) -> list[Future]:
        """
        Batch futures from the iterable into tuples of length n. The last batch may be shorter than n.

        Args:
            iterable (list): list of future objects to batch based on which future objects finish first
            n (int): batch size

        Returns:
            list[Future]: list of future objects one for each batch
        """
        skip_lst: list[Future] = []
        future_lst: list[Future] = []
        for _ in range(len(iterable) // n + (1 if len(iterable) % n > 0 else 0)):
            f: Future = Future()
            if self._future_queue is not None:
                self._future_queue.put(
                    {
                        "fn": "batched",
                        "args": (),
                        "kwargs": {"lst": iterable, "n": n, "skip_lst": skip_lst},
                        "future": f,
                        "resource_dict": {},
                    }
                )
            skip_lst = skip_lst.copy() + [f]  # be careful
            future_lst.append(f)

        return future_lst

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """
        Exit method called when exiting the context manager.

        Args:
            exc_type: The type of the exception.
            exc_val: The exception instance.
            exc_tb: The traceback object.

        """
        super().__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)  # type: ignore
        if self._generate_dependency_graph:
            node_lst, edge_lst = generate_nodes_and_edges_for_plotting(
                task_hash_dict=self._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            return plot_dependency_graph_function(
                node_lst=node_lst,
                edge_lst=edge_lst,
                filename=self._plot_dependency_graph_filename,
            )
        else:
            return None


def _execute_tasks_with_dependencies(
    future_queue: queue.Queue,
    executor_queue: queue.Queue,
    executor: TaskSchedulerBase,
    refresh_rate: float = 0.01,
):
    """
    Resolve the dependencies of multiple tasks, by analysing which task requires concurrent.future.Futures objects from
    other tasks.

    Args:
        future_queue (Queue): Queue for receiving new tasks.
        executor_queue (Queue): Queue for the internal executor.
        executor (TaskSchedulerBase): Executor to execute the tasks with after the dependencies are resolved.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
    """
    wait_lst = []
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            task_dict = None
        if (  # shutdown the executor
            task_dict is not None and "shutdown" in task_dict and task_dict["shutdown"]
        ):
            executor.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        if (  # shutdown the executor
            task_dict is not None and "internal" in task_dict and task_dict["internal"]
        ):
            if task_dict["task"] == "get_info":
                task_dict["future"].set_result(executor.info)
            elif task_dict["task"] == "get_max_workers":
                task_dict["future"].set_result(executor.max_workers)
            elif task_dict["task"] == "set_max_workers":
                try:
                    executor.max_workers = task_dict["max_workers"]
                except NotImplementedError:
                    task_dict["future"].set_result(False)
                else:
                    task_dict["future"].set_result(True)
        elif (  # handle function submitted to the executor
            task_dict is not None and "fn" in task_dict and "future" in task_dict
        ):
            future_lst, ready_flag = get_future_objects_from_input(
                args=task_dict["args"], kwargs=task_dict["kwargs"]
            )
            exception_lst = get_exception_lst(future_lst=future_lst)
            if not check_exception_was_raised(future_obj=task_dict["future"]):
                if len(exception_lst) > 0:
                    task_dict["future"].set_exception(exception_lst[0])
                elif len(future_lst) == 0 or ready_flag:
                    # No future objects are used in the input or all future objects are already done
                    task_dict["args"], task_dict["kwargs"] = update_futures_in_input(
                        args=task_dict["args"], kwargs=task_dict["kwargs"]
                    )
                    executor_queue.put(task_dict)
                else:  # Otherwise add the function to the wait list
                    task_dict["future_lst"] = future_lst
                    wait_lst.append(task_dict)
            future_queue.task_done()
        elif len(wait_lst) > 0:
            number_waiting = len(wait_lst)
            # Check functions in the wait list and execute them if all future objects are now ready
            wait_lst = _update_waiting_task(
                wait_lst=wait_lst, executor_queue=executor_queue
            )
            # if no job is ready, sleep for a moment
            if len(wait_lst) == number_waiting:
                sleep(refresh_rate)
        else:
            # If there is nothing else to do, sleep for a moment
            sleep(refresh_rate)


def _update_waiting_task(wait_lst: list[dict], executor_queue: queue.Queue) -> list:
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
        exception_lst = get_exception_lst(future_lst=task_wait_dict["future_lst"])
        if len(exception_lst) > 0:
            task_wait_dict["future"].set_exception(exception_lst[0])
        elif task_wait_dict["fn"] != "batched" and all(
            future.done() for future in task_wait_dict["future_lst"]
        ):
            del task_wait_dict["future_lst"]
            task_wait_dict["args"], task_wait_dict["kwargs"] = update_futures_in_input(
                args=task_wait_dict["args"], kwargs=task_wait_dict["kwargs"]
            )
            executor_queue.put(task_wait_dict)
        elif task_wait_dict["fn"] == "batched" and all(
            future.done() for future in task_wait_dict["kwargs"]["skip_lst"]
        ):
            done_lst = batched_futures(
                lst=task_wait_dict["kwargs"]["lst"],
                n=task_wait_dict["kwargs"]["n"],
                skip_lst=[f.result() for f in task_wait_dict["kwargs"]["skip_lst"]],
            )
            if len(done_lst) == 0:
                wait_tmp_lst.append(task_wait_dict)
            else:
                task_wait_dict["future"].set_result(done_lst)
        else:
            wait_tmp_lst.append(task_wait_dict)
    return wait_tmp_lst
