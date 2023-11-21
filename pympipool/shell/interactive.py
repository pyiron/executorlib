from concurrent.futures import Future
import subprocess
from time import sleep

from pympipool.shared.executorbase import cancel_items_in_queue, ExecutorBase
from pympipool.shared.thread import RaisingThread


def wait_for_process_to_stop(process, sleep_interval=10e-10):
    while process.poll() is None:
        sleep(sleep_interval)


def execute_single_task(future_queue):
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


class ShellInteractiveExecutor(ExecutorBase):
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
        if cancel_futures:
            cancel_items_in_queue(que=self._future_queue)
        self._future_queue.put({"shutdown": True, "wait": wait})
        if wait:
            self._process.join()
            # self._future_queue.join()
        self._process = None
        self._future_queue = None
