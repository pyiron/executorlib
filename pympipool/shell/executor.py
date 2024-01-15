from concurrent.futures import Future
import subprocess

from pympipool.shared.executorbase import executor_broker, ExecutorBase
from pympipool.shared.thread import RaisingThread


def execute_single_task(future_queue):
    """
    Process items received via the queue.

    Args:
        future_queue (queue.Queue):
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


class SubprocessSingleExecutor(ExecutorBase):
    """
    The pympipool.shell.SubprocessSingleExecutor is the internal worker for the pympipool.shell.SubprocessExecutor.
    """

    def __init__(self):
        super().__init__()
        self._process = RaisingThread(
            target=execute_single_task,
            kwargs={
                "future_queue": self._future_queue,
            },
        )
        self._process.start()

    def submit(self, *args, **kwargs):
        f = Future()
        self._future_queue.put({"future": f, "args": args, "kwargs": kwargs})
        return f


class SubprocessExecutor(ExecutorBase):
    """
    The pympipool.shell.SubprocessExecutor enables the submission of command line calls via the subprocess.check_output()
    interface of the python standard library. It is based on the concurrent.futures.Executor class and returns a
    concurrent.futures.Future object for every submitted command line call. Still it does not provide any option to
    interact with the external executable during the execution.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel

    Examples:

        >>> from pympipool import SubprocessExecutor
        >>> with SubprocessExecutor(max_workers=2) as exe:
        >>>     future = exe.submit(["echo", "test"], universal_newlines=True)
        >>> print(future.done(), future.result(), future.done())
        (False, "test", True)

    """

    def __init__(
        self,
        max_workers=1,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                # Broker Arguments
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "executor_class": SubprocessSingleExecutor,
            },
        )
        self._process.start()

    def submit(self, *args, **kwargs):
        """
        Submit a command line call to be executed. The given arguments are provided to subprocess.Popen() as additional
        inputs to control the execution.

        Returns:
            A Future representing the given call.
        """
        f = Future()
        self._future_queue.put({"future": f, "args": args, "kwargs": kwargs})
        return f
