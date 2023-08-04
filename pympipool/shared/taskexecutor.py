import inspect
import os
import queue
import sys

import cloudpickle

from pympipool.shared.communication import interface_bootup


def cancel_items_in_queue(que):
    """
    Cancel items which are still waiting in the queue. If the executor is busy tasks remain in the queue, so the future
    objects have to be cancelled when the executor shuts down.

    Args:
        que (queue.Queue): Queue with task objects which should be executed
    """
    while True:
        try:
            item = que.get_nowait()
            if isinstance(item, dict) and "future" in item.keys():
                item["future"].cancel()
                que.task_done()
        except queue.Empty:
            break


def cloudpickle_register(ind=2):
    """
    Cloudpickle can either pickle by value or pickle by reference. The functions which are communicated have to
    be pickled by value rather than by reference, so the module which calls the map function is pickled by value.
    https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
    inspect can help to find the module which is calling pympipool
    https://docs.python.org/3/library/inspect.html
    to learn more about inspect another good read is:
    http://pymotw.com/2/inspect/index.html#module-inspect
    1 refers to 1 level higher than the map function

    Args:
        ind (int): index of the level at which pickle by value starts while for the rest pickle by reference is used
    """
    try:  # When executed in a jupyter notebook this can cause a ValueError - in this case we just ignore it.
        cloudpickle.register_pickle_by_value(inspect.getmodule(inspect.stack()[ind][0]))
    except IndexError:
        cloudpickle_register(ind=ind - 1)
    except ValueError:
        pass


def execute_parallel_tasks(
    future_queue,
    cores,
    threads_per_core=1,
    gpus_per_task=0,
    cwd=None,
    executor=None,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       threads_per_core (int): number of OpenMP threads to be used for each function call
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
       cwd (str/None): current working directory where the parallel python task is executed
       executor (flux.job.FluxExecutor/None): flux executor to submit tasks to - optional
    """
    command_lst = [sys.executable]
    if cores > 1:
        command_lst += [
            os.path.abspath(
                os.path.join(__file__, "..", "..", "backend", "mpiexec.py")
            ),
        ]
    else:
        command_lst += [
            os.path.abspath(os.path.join(__file__, "..", "..", "backend", "serial.py")),
        ]
    interface = interface_bootup(
        command_lst=command_lst,
        cwd=cwd,
        cores=cores,
        threads_per_core=threads_per_core,
        gpus_per_core=gpus_per_task,
        executor=executor,
    )
    execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def execute_parallel_tasks_loop(interface, future_queue):
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
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
        elif "fn" in task_dict.keys() and "init" in task_dict.keys():
            interface.send_dict(input_dict=task_dict)
            future_queue.task_done()
