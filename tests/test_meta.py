from concurrent.futures import as_completed, Future, ThreadPoolExecutor
import unittest
from pympipool.external_interfaces.meta import _get_future_done, MetaExecutorFuture, MetaExecutor


def calc(i):
    return i


class TestFutureCreation(unittest.TestCase):
    def test_get_future_done(self):
        f = _get_future_done()
        self.assertTrue(isinstance(f, Future))
        self.assertTrue(f.done())


class TestMetaExecutorFuture(unittest.TestCase):
    def test_meta_executor_future(self):
        meta_future = MetaExecutorFuture(future=_get_future_done(), executor=ThreadPoolExecutor(max_workers=1))
        self.assertTrue(isinstance(meta_future.future, Future))
        self.assertTrue(isinstance(meta_future.executor, ThreadPoolExecutor))
        self.assertTrue(meta_future.future.done())
        self.assertEqual(meta_future, next(as_completed([meta_future])))


class TestMetaExecutor(unittest.TestCase):
    def test_meta_executor(self):
        with MetaExecutor(max_workers=2, cores_per_worker=1, sleep_interval=0.1) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())
