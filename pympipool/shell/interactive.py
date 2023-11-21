from concurrent.futures import Future
import subprocess
from time import sleep

from pympipool.shared.executorbase import cancel_items_in_queue, ExecutorBase
from pympipool.shared.thread import RaisingThread


def wait_for_process_to_stop(process, sleep_interval=10e-10):
    """
    Wait for the subprocess.Popen() process to stop executing

    Args:
        process (subprocess.Popen): process object
        sleep_interval (float): interval to sleep during poll() calls
    """
    while process.poll() is None:
        sleep(sleep_interval)


def execute_single_task(future_queue):
    """
    Process items received via the queue.

    Args:
        future_queue (queue.Queue):
    """
    process = None
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            if process is not None and process.poll() is None:
                process.stdin.flush()
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()
                process.terminate()
                wait_for_process_to_stop(process=process)
            future_queue.task_done()
            # future_queue.join()
            break
        elif "init" in task_dict.keys() and task_dict["init"]:
            process = subprocess.Popen(
                *task_dict["args"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **task_dict["kwargs"],
            )
        elif "future" in task_dict.keys():
            if process is None:
                raise ValueError("process not initialized")
            elif process.poll() is None:
                f = task_dict.pop("future")
                if f.set_running_or_notify_cancel():
                    try:
                        process.stdin.write(task_dict["input"])
                        process.stdin.flush()
                        lines_count = 0
                        output = ""
                        while True:
                            output_current = process.stdout.readline()
                            output += output_current
                            lines_count += 1
                            if (
                                task_dict["stop_read_pattern"] is not None
                                and task_dict["stop_read_pattern"] in output_current
                            ):
                                break
                            elif (
                                task_dict["lines_to_read"] is not None
                                and task_dict["lines_to_read"] == lines_count
                            ):
                                break
                        f.set_result(output)
                    except Exception as thread_exception:
                        future_queue.task_done()
                        f.set_exception(exception=thread_exception)
                        raise thread_exception
                    else:
                        future_queue.task_done()
            else:
                raise ValueError("process exited")


class ShellExecutor(ExecutorBase):
    """
    In contrast to the other pympipool.shell.SubprocessExecutor and the pympipool.Executor the pympipool.shell.ShellExecutor
    can only execute a single process at a given time. Still it adds the capability to interact with this process during
    its execution. The initialization of the pympipool.shell.ShellExecutor takes the same input arguments as the
    subprocess.Popen() call for the standard library to start a subprocess.

    Examples

        >>> from pympipool import ShellExecutor
        >>> with ShellExecutor(["python", "count.py"], universal_newlines=True) as exe:
        >>>     future_lines = exe.submit(string_input="4", lines_to_read=5)
        >>>     print(future_lines.done(), future_lines.result(), future_lines.done())
        (False, "0\n1\n2\n3\ndone\n", True)

        >>> from pympipool import ShellExecutor
        >>> with ShellExecutor(["python", "count.py"], universal_newlines=True) as exe:
        >>>     future_pattern = exe.submit(string_input="4", stop_read_pattern="done")
        >>>     print(future_pattern.done(), future_pattern.result(), future_pattern.done())
        (False, "0\n1\n2\n3\ndone\n", True)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._process = RaisingThread(
            target=execute_single_task,
            kwargs={
                "future_queue": self._future_queue,
            },
        )
        self._process.start()
        self._future_queue.put({"init": True, "args": args, "kwargs": kwargs})

    def submit(self, string_input, lines_to_read=None, stop_read_pattern=None):
        """
        Submit the input as a string to the executable. In addition to the input the ShellExecutor also needs a measure
        to identify the completion of the execution. This can either be provided based on the number of lines to read
        using the `lines_to_read` parameter or by providing a string pattern using the `stop_read_pattern` to stop
        reading new lines. One of these two stopping criteria has to be defined.

        Args:
            string_input (str): Input to be communicated to the underlying executable
            lines_to_read (None/int): integer number of lines to read from the command line (optional)
            stop_read_pattern (None/str): string pattern to indicate the command line output is completed (optional)

        Returns:
            A Future representing the given call.
        """
        if lines_to_read is None and stop_read_pattern is None:
            raise ValueError(
                "Either the number of lines_to_read (int) or the stop_read_pattern (str) has to be defined."
            )
        if string_input[-1:] != "\n":
            string_input += "\n"
        f = Future()
        self._future_queue.put(
            {
                "future": f,
                "input": string_input,
                "lines_to_read": lines_to_read,
                "stop_read_pattern": stop_read_pattern,
            }
        )
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
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
        self._future_queue.put({"shutdown": True, "wait": wait})
        if wait:
            self._process.join()
            # self._future_queue.join()
        self._process = None
        self._future_queue = None
