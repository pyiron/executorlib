import asyncio
import os
import threading
import time
import unittest
from concurrent.futures import CancelledError, Future
from unittest.mock import patch

from executorlib import SingleNodeExecutor
from executorlib.standalone.interactive.spawner import MpiExecSpawner
from executorlib.task_scheduler.interactive.blockallocation import (
    BlockAllocationTaskScheduler,
)
from executorlib.task_scheduler.interactive.blockallocation_async import (
    AsyncWorker,
)


def add(a, b):
    return a + b


def sleep_and_return(x, seconds=0.2):
    import time

    time.sleep(seconds)
    return x


def raise_error(msg):
    raise ValueError(msg)


def get_k(k):
    return k


def init_k():
    return {"k": 42}


def _scheduler(max_workers=1, **executor_kwargs):
    return BlockAllocationTaskScheduler(
        max_workers=max_workers,
        executor_kwargs=executor_kwargs | {"hostname_localhost": True},
        spawner=MpiExecSpawner,
    )


@patch.dict(os.environ, {"EXECUTORLIB_ASYNCIO": "1"})
class TestAsyncBlockAllocation(unittest.TestCase):
    def test_uses_async_workers(self):
        with _scheduler(max_workers=2) as exe:
            self.assertTrue(all(isinstance(p, AsyncWorker) for p in exe._process))
            self.assertEqual(exe.submit(add, 1, 2).result(), 3)

    def test_returns_concurrent_future(self):
        with _scheduler() as exe:
            f = exe.submit(add, 1, b=2)
            self.assertIsInstance(f, Future)
            self.assertEqual(f.result(), 3)
            self.assertTrue(f.done())

    def test_many_simultaneous_tasks(self):
        with _scheduler(max_workers=4) as exe:
            fs = [exe.submit(sleep_and_return, i, seconds=0.1) for i in range(16)]
            self.assertEqual([f.result() for f in fs], list(range(16)))

    def test_parallel_execution(self):
        with _scheduler(max_workers=4) as exe:
            exe.submit(add, 0, 0).result()  # wait for all workers to boot
            start = time.time()
            fs = [exe.submit(sleep_and_return, i, seconds=1.0) for i in range(4)]
            [f.result() for f in fs]
            self.assertLess(time.time() - start, 3.0)

    def test_exception(self):
        with _scheduler(max_workers=2) as exe:
            f = exe.submit(raise_error, "boom")
            with self.assertRaises(ValueError):
                f.result()
            # the worker stays usable after an exception
            self.assertEqual(exe.submit(add, 2, 2).result(), 4)

    def test_init_function(self):
        with _scheduler(max_workers=2, init_function=init_k) as exe:
            self.assertEqual(exe.submit(get_k).result(), 42)

    def test_shutdown_wait(self):
        exe = _scheduler(max_workers=2)
        fs = [exe.submit(sleep_and_return, i) for i in range(4)]
        pool = exe._async_pool
        exe.shutdown(wait=True)
        self.assertTrue(all(f.done() for f in fs))
        self.assertEqual([f.result() for f in fs], list(range(4)))
        self.assertFalse(pool._loop_thread.is_alive())
        self.assertFalse(pool._dispatch_thread.is_alive())
        self.assertTrue(pool._loop.is_closed())

    def test_shutdown_no_wait(self):
        exe = _scheduler(max_workers=1)
        f = exe.submit(sleep_and_return, 1)
        pool = exe._async_pool
        exe.shutdown(wait=False)
        self.assertEqual(f.result(), 1)
        pool._loop_thread.join(timeout=30)
        pool._dispatch_thread.join(timeout=30)
        self.assertFalse(pool._loop_thread.is_alive())
        self.assertFalse(pool._dispatch_thread.is_alive())

    def test_shutdown_cancel_futures(self):
        exe = _scheduler(max_workers=1)
        fs = [exe.submit(sleep_and_return, i, seconds=0.5) for i in range(6)]
        fs[0].result()
        exe.shutdown(wait=True, cancel_futures=True)
        self.assertTrue(any(f.cancelled() for f in fs))
        for f in fs:
            if f.cancelled():
                with self.assertRaises(CancelledError):
                    f.result()

    def test_repeated_creation(self):
        base = threading.active_count()
        for i in range(5):
            with _scheduler(max_workers=2) as exe:
                self.assertEqual(exe.submit(add, i, 1).result(), i + 1)
        self.assertEqual(threading.active_count(), base)

    def test_resize(self):
        with _scheduler(max_workers=1) as exe:
            exe.max_workers = 3
            self.assertEqual(len(exe._process), 3)
            fs = [exe.submit(add, i, 1) for i in range(6)]
            self.assertEqual([f.result() for f in fs], [i + 1 for i in range(6)])
            exe.max_workers = 1
            self.assertEqual(len(exe._process), 1)
            self.assertEqual(exe.submit(add, 1, 1).result(), 2)

    def test_single_node_executor(self):
        with SingleNodeExecutor(max_workers=2, block_allocation=True) as exe:
            fs = [exe.submit(add, i, i) for i in range(8)]
            self.assertEqual([f.result() for f in fs], [2 * i for i in range(8)])


@patch.dict(os.environ, {"EXECUTORLIB_ASYNCIO": "1"})
class TestAsyncRunningEventLoop(unittest.TestCase):
    """
    Simulate Jupyter/IPython, where the calling thread already runs an asyncio event loop.
    """

    def test_sync_api_inside_running_loop(self):
        async def existing_application():
            running_loop = asyncio.get_running_loop()
            with _scheduler(max_workers=2) as exe:
                self.assertIsNot(exe._async_pool._loop, running_loop)
                fs = [exe.submit(add, i, 1) for i in range(4)]
                result = [f.result() for f in fs]
            self.assertIs(asyncio.get_running_loop(), running_loop)
            return result

        self.assertEqual(asyncio.run(existing_application()), [1, 2, 3, 4])

    def test_await_wrapped_future_inside_running_loop(self):
        async def existing_application():
            with SingleNodeExecutor(max_workers=2, block_allocation=True) as exe:
                fs = [exe.submit(sleep_and_return, i, seconds=0.1) for i in range(4)]
                # the running loop stays responsive while executorlib works
                ticks = 0
                while not all(f.done() for f in fs):
                    await asyncio.sleep(0.01)
                    ticks += 1
                result = await asyncio.gather(*[asyncio.wrap_future(f) for f in fs])
            return result, ticks

        result, ticks = asyncio.run(existing_application())
        self.assertEqual(result, [0, 1, 2, 3])
        self.assertGreater(ticks, 0)

    def test_exception_inside_running_loop(self):
        async def existing_application():
            with _scheduler() as exe:
                with self.assertRaises(ValueError):
                    await asyncio.wrap_future(exe.submit(raise_error, "boom"))

        asyncio.run(existing_application())


class TestAsyncThreadScaling(unittest.TestCase):
    @staticmethod
    def _admin_threads(max_workers):
        base = threading.active_count()
        with _scheduler(max_workers=max_workers) as exe:
            fs = [exe.submit(add, i, 1) for i in range(max_workers)]
            [f.result() for f in fs]
            during = threading.active_count() - base
        return during

    def test_thread_count_constant(self):
        with patch.dict(os.environ, {"EXECUTORLIB_ASYNCIO": "1"}):
            counts = {n: self._admin_threads(n) for n in (1, 4, 16)}
        # event loop thread + dispatcher thread, independent of max_workers
        self.assertEqual(counts[1], counts[16], counts)
        self.assertLessEqual(counts[16], 3, counts)

    def test_thread_count_threaded_reference(self):
        with patch.dict(os.environ, {"EXECUTORLIB_ASYNCIO": "0"}):
            counts = {n: self._admin_threads(n) for n in (1, 4, 16)}
        self.assertEqual(counts[16] - counts[1], 15, counts)


if __name__ == "__main__":
    unittest.main()
