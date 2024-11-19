import unittest

from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register


def calc(i):
    return i


def resource_dict(resource_dict):
    return resource_dict


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial_with_dependencies(self):
        with Executor(
            max_cores=2,
            backend="local",
            block_allocation=False,
            disable_dependencies=True,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_serial_without_dependencies(self):
        with Executor(
            max_cores=2,
            backend="local",
            block_allocation=False,
            disable_dependencies=False,
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
                resource_dict={
                    "cores": 1,
                    "gpus_per_core": 1,
                },
                backend="local",
            )
        with self.assertRaises(ValueError):
            with Executor(
                max_cores=1,
                backend="local",
                block_allocation=False,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})
        with self.assertRaises(ValueError):
            with Executor(
                max_cores=1,
                backend="local",
                block_allocation=True,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})
