from concurrent.futures import Future
from typing import Any, Callable, Dict

from executorlib.interactive import create_executor
from executorlib.shared.executor import ExecutorSteps, execute_tasks_with_dependencies
from executorlib.shared.plot import draw, generate_nodes_and_edges, generate_task_hash
from executorlib.shared.thread import RaisingThread


class ExecutorWithDependencies(ExecutorSteps):
    """
    ExecutorWithDependencies is a class that extends ExecutorSteps and provides
    functionality for executing tasks with dependencies.

    Args:
        refresh_rate (float, optional): The refresh rate for updating the executor queue. Defaults to 0.01.
        plot_dependency_graph (bool, optional): Whether to generate and plot the dependency graph. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _future_hash_dict (Dict[str, Future]): A dictionary mapping task hash to future object.
        _task_hash_dict (Dict[str, Dict]): A dictionary mapping task hash to task dictionary.
        _generate_dependency_graph (bool): Whether to generate the dependency graph.

    """

    def __init__(
        self,
        *args: Any,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        executor = create_executor(*args, **kwargs)
        self._set_process(
            RaisingThread(
                target=execute_tasks_with_dependencies,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "executor_queue": executor._future_queue,
                    "executor": executor,
                    "refresh_rate": refresh_rate,
                },
            )
        )
        self._future_hash_dict = {}
        self._task_hash_dict = {}
        self._generate_dependency_graph = plot_dependency_graph

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        resource_dict: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Future:
        """
        Submits a task to the executor.

        Args:
            fn (callable): The function to be executed.
            *args: Variable length argument list.
            resource_dict (dict, optional): A dictionary of resources required by the task. Defaults to {}.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Future: A future object representing the result of the task.

        """
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
            task_hash = generate_task_hash(
                task_dict=task_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            self._future_hash_dict[task_hash] = f
            self._task_hash_dict[task_hash] = task_dict
        return f

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
        super().__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)
        if self._generate_dependency_graph:
            node_lst, edge_lst = generate_nodes_and_edges(
                task_hash_dict=self._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            return draw(node_lst=node_lst, edge_lst=edge_lst)
