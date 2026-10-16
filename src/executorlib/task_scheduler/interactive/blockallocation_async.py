"""
Experimental asyncio control plane for the BlockAllocationTaskScheduler.

Instead of one Python thread per worker, all workers of an executor are driven by coroutines on a single private
asyncio event loop, which runs in one dedicated background thread owned by the executor. The ZMQ communication uses
zmq.asyncio. The task queue stays a queue.Queue; a single dispatcher thread hands its items to idle worker coroutines,
so tasks remain in the queue (and can be cancelled) until a worker is ready, exactly like in the threaded version.

Resulting administrative threads per executor: 2 (event loop + dispatcher), independent of the number of workers.

The event loop of the calling thread (e.g. the one of a Jupyter kernel) is never used or modified.
"""

import asyncio
import queue
import traceback
from threading import Event, Lock, Thread
from typing import Callable, Optional

import zmq.asyncio

from executorlib.standalone.command import get_interactive_execute_command
from executorlib.standalone.interactive.communication import (
    AsyncSocketInterface,
    ExecutorlibSocketError,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.task_scheduler.interactive.shared import (
    execute_task_dict_async,
    reset_task_dict,
    task_done,
)


class AsyncWorkerPool:
    """
    Private asyncio event loop in one background thread, plus one dispatcher thread which blocks on the shared
    queue.Queue on behalf of the idle worker coroutines.

    Args:
        future_queue (queue.Queue): task queue shared with the executor
    """

    def __init__(self, future_queue: queue.Queue):
        self._future_queue = future_queue
        self._loop = asyncio.new_event_loop()
        self.context = zmq.asyncio.Context()
        self._idle_workers: queue.Queue = queue.Queue()
        self._lock = Lock()
        self._active_workers = 0
        self._closing = False
        self._close_requested = False
        self._loop_thread = Thread(target=self._run_loop)
        self._dispatch_thread = Thread(target=self._dispatch)
        self._loop_thread.start()
        self._dispatch_thread.start()

    def start_worker(self, worker: "AsyncWorker"):
        with self._lock:
            self._active_workers += 1
        asyncio.run_coroutine_threadsafe(self._run_worker(worker), self._loop)

    def close(self, wait: bool = True):
        """
        Stop the event loop and the dispatcher thread as soon as all worker coroutines have finished.

        Args:
            wait (bool): block until both threads are stopped
        """
        if not self._close_requested:
            self._close_requested = True
            self._loop.call_soon_threadsafe(self._request_close)
        if wait:
            self._loop_thread.join()
            self._dispatch_thread.join()

    async def get_task(self) -> dict:
        """
        Wait for the next item of the shared queue.Queue without blocking the event loop.
        """
        waiter = self._loop.create_future()
        self._idle_workers.put(waiter)
        return await waiter

    def _dispatch(self):
        while True:
            waiter = self._idle_workers.get()
            if waiter is None:
                break
            task_dict = self._future_queue.get()
            self._loop.call_soon_threadsafe(waiter.set_result, task_dict)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self.context.destroy(linger=0)
        self._loop.close()

    async def _run_worker(self, worker: "AsyncWorker"):
        try:
            await _execute_multiple_tasks_async(pool=self, **worker.kwargs)
        except Exception:
            traceback.print_exc()
        finally:
            worker.done.set()
            with self._lock:
                self._active_workers -= 1
            self._stop_if_finished()

    def _request_close(self):
        self._closing = True
        self._stop_if_finished()

    def _stop_if_finished(self):
        with self._lock:
            finished = self._closing and self._active_workers == 0
        if finished:
            self._idle_workers.put(None)
            self._loop.stop()


class AsyncWorker:
    """
    Thread-like handle (start(), join(), is_alive()) for a worker coroutine, so the BlockAllocationTaskScheduler can
    manage it like the Thread objects it uses otherwise.
    """

    def __init__(self, pool: AsyncWorkerPool, kwargs: dict):
        self._pool = pool
        self.kwargs = kwargs
        self.done = Event()

    def start(self):
        self._pool.start_worker(self)

    def join(self, timeout: Optional[float] = None):
        self.done.wait(timeout)

    def is_alive(self) -> bool:
        return not self.done.is_set()


async def _execute_multiple_tasks_async(
    pool: AsyncWorkerPool,
    future_queue: queue.Queue,
    cores: int = 1,
    spawner: type[BaseSpawner] = MpiExecSpawner,
    hostname_localhost: Optional[bool] = None,
    init_function: Optional[Callable] = None,
    cache_directory: Optional[str] = None,
    cache_key: Optional[str] = None,
    log_obj_size: bool = False,
    error_log_file: Optional[str] = None,
    worker_id: int = 0,
    stop_function: Optional[Callable] = None,
    restart_limit: int = 0,
    next_bootup_event: Optional[Event] = None,
    alive_workers: Optional[list] = None,
    alive_workers_lock: Optional[Lock] = None,
    bootup_event: Optional[Event] = None,
    queue_join_on_shutdown: bool = False,
    **kwargs,
) -> None:
    """
    Coroutine version of blockallocation._execute_multiple_tasks(), see there for the arguments.

    The worker coroutines are started in worker_id order and boot up their process synchronously before their first
    await, so the boot order is preserved without waiting on bootup_event (which would block the event loop). For the
    same reason queue_join_on_shutdown is not supported; the BlockAllocationTaskScheduler always sets it to False.
    """
    interface = interface_bootup(
        command_lst=get_interactive_execute_command(
            cores=cores,
        ),
        connections=spawner(cores=cores, worker_id=worker_id, **kwargs),
        hostname_localhost=hostname_localhost,
        log_obj_size=log_obj_size,
        worker_id=worker_id,
        stop_function=stop_function,
        context=pool.context,
    )
    assert isinstance(interface, AsyncSocketInterface)
    if next_bootup_event is not None:
        next_bootup_event.set()
    interface_initialization_exception = await _set_init_function_async(
        interface=interface,
        init_function=init_function,
    )
    restart_counter = 0
    while True:
        if not interface.status and restart_counter >= restart_limit:
            await _drain_dead_worker_async(
                pool=pool,
                future_queue=future_queue,
                alive_workers=alive_workers,
                alive_workers_lock=alive_workers_lock,
            )
            break
        elif not interface.status:
            interface.bootup()
            interface_initialization_exception = await _set_init_function_async(
                interface=interface,
                init_function=init_function,
            )
            restart_counter += 1
        else:  # interface.status == True
            task_dict = await pool.get_task()
            if "shutdown" in task_dict and task_dict["shutdown"]:
                if interface.status:
                    await interface.shutdown_async(wait=task_dict["wait"])
                task_done(future_queue=future_queue)
                break
            elif "fn" in task_dict and "future" in task_dict:
                f = task_dict.pop("future")
                if interface_initialization_exception is not None:
                    f.set_exception(exception=interface_initialization_exception)
                else:
                    # The interface failed during the execution
                    interface.status = await execute_task_dict_async(
                        task_dict=task_dict,
                        future_obj=f,
                        interface=interface,
                        cache_directory=cache_directory,
                        cache_key=cache_key,
                        error_log_file=error_log_file,
                    )
                    if not interface.status:
                        reset_task_dict(
                            future_obj=f, future_queue=future_queue, task_dict=task_dict
                        )
                task_done(future_queue=future_queue)


async def _drain_dead_worker_async(
    pool: AsyncWorkerPool,
    future_queue: queue.Queue,
    alive_workers: Optional[list] = None,
    alive_workers_lock: Optional[Lock] = None,
) -> None:
    """
    Coroutine version of blockallocation._drain_dead_worker().
    """
    if alive_workers is not None and alive_workers_lock is not None:
        with alive_workers_lock:
            if alive_workers[0] > 0:
                alive_workers[0] -= 1
    while True:
        task_dict = await pool.get_task()
        if "shutdown" in task_dict and task_dict["shutdown"]:
            task_done(future_queue=future_queue)
            break
        elif "fn" in task_dict and "future" in task_dict:
            if alive_workers is not None and alive_workers_lock is not None:
                with alive_workers_lock:
                    has_healthy_workers = alive_workers[0] > 0
            else:
                has_healthy_workers = False
            if has_healthy_workers:
                future_queue.put(task_dict)
                task_done(future_queue=future_queue)
                # give the healthy workers a chance to pick up the recycled task
                await asyncio.sleep(0.01)
            else:
                f = task_dict.pop("future")
                f.set_exception(
                    ExecutorlibSocketError("SocketInterface crashed during execution.")
                )
                task_done(future_queue=future_queue)


async def _set_init_function_async(
    interface: AsyncSocketInterface,
    init_function: Optional[Callable] = None,
) -> Optional[Exception]:
    interface_initialization_exception = None
    if init_function is not None and interface.status:
        output = await interface.send_and_receive_dict_async(
            input_dict={"init": True, "fn": init_function, "args": (), "kwargs": {}}
        )
        if "error" in output:
            interface_initialization_exception = output["error"]
    return interface_initialization_exception
