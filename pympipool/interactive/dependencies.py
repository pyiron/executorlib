from concurrent.futures import Future

from pympipool.interactive import create_executor
from pympipool.shared.executor import ExecutorSteps, execute_tasks_with_dependencies
from pympipool.shared.thread import RaisingThread


try:
    from pympipool.shared.plot import generate_task_hash, generate_nodes_and_edges, draw

    graph_extension_available = True
    import_error = None
except ImportError:
    graph_extension_available = False


class ExecutorWithDependencies(ExecutorSteps):
    def __init__(
        self,
        *args,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        **kwargs,
    ):
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
        if plot_dependency_graph and not graph_extension_available:
            raise ImportError(import_error)
        self._generate_dependency_graph = plot_dependency_graph

    def submit(self, fn: callable, *args, resource_dict: dict = {}, **kwargs):
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)
        if self._generate_dependency_graph:
            node_lst, edge_lst = generate_nodes_and_edges(
                task_hash_dict=self._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            return draw(node_lst=node_lst, edge_lst=edge_lst)
