import unittest

from executorlib import Executor
from executorlib.shared.executor import cloudpickle_register


def calc(i):
    return i


def resource_dict(resource_dict):
    return resource_dict


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial(self):
        with Executor(
            max_cores=2,
            hostname_localhost=True,
            backend="local",
            block_allocation=False,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_single(self):
        with Executor(
            max_cores=1,
            hostname_localhost=True,
            backend="local",
            block_allocation=False,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_errors(self):
        with self.assertRaises(TypeError):
            Executor(
                max_cores=1,
                cores_per_worker=1,
                threads_per_core=2,
                hostname_localhost=True,
                backend="local",
            )
        with self.assertRaises(TypeError):
            Executor(
                max_cores=1,
                cores_per_worker=1,
                gpus_per_worker=1,
                hostname_localhost=True,
                backend="local",
            )
        with self.assertRaises(ValueError):
            with Executor(
                max_cores=1,
                hostname_localhost=True,
                backend="local",
                block_allocation=False,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})
        with self.assertRaises(ValueError):
            with Executor(
                max_cores=1,
                hostname_localhost=True,
                backend="local",
                block_allocation=True,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})
