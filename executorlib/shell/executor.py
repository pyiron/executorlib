import queue
import subprocess
from concurrent.futures import Future
from typing import Any

from executorlib.shared.executor import ExecutorBroker
from executorlib.shared.thread import RaisingThread


def execute_single_task(
    future_queue: queue.Queue,
) -> None:
    """
    Process items received via the queue.

    Args:
        future_queue (queue.Queue): The queue containing the tasks to be executed.
    """
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            future_queue.task_done()
            future_queue.join()
            break
        elif "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(
                        subprocess.check_output(
                            *task_dict["args"], **task_dict["kwargs"]
                        )
                    )
                except Exception as thread_exception:
                    future_queue.task_done()
                    f.set_exception(exception=thread_exception)
                    raise thread_exception
                else:
                    future_queue.task_done()
        else:
            raise KeyError(task_dict)


class SubprocessExecutor(ExecutorBroker):
    """
    The executorlib.shell.SubprocessExecutor enables the submission of command line calls via the subprocess.check_output()
    interface of the python standard library. It is based on the concurrent.futures.Executor class and returns a
    concurrent.futures.Future object for every submitted command line call. Still it does not provide any option to
    interact with the external executable during the execution.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel

    Examples:

        >>> from executorlib import SubprocessExecutor
        >>> with SubprocessExecutor(max_workers=2) as exe:
        >>>     future = exe.submit(["echo", "test"], universal_newlines=True)
        >>> print(future.done(), future.result(), future.done())
        (False, "test", True)

    """

    def __init__(
        self,
        max_workers: int = 1,
    ):
        super().__init__()
        self._set_process(
            process=[
                RaisingThread(
                    target=execute_single_task,
                    kwargs={
                        # Executor Arguments
                        "future_queue": self._future_queue,
                    },
                )
                for _ in range(max_workers)
            ],
        )

    def submit(self, *args: Any, **kwargs: Any) -> Future:
        """
        Submit a command line call to be executed. The given arguments are provided to subprocess.Popen() as additional
        inputs to control the execution.

        Args:
            *args: Additional arguments to be passed to subprocess.Popen().
            **kwargs: Additional keyword arguments to be passed to subprocess.Popen().

        Returns:
            A Future representing the given call.
        """
        f: Future = Future()
        self._future_queue.put({"future": f, "args": args, "kwargs": kwargs})
        return f
