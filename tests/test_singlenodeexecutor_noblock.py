import unittest
from time import sleep

from executorlib import SingleNodeExecutor
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.standalone.interactive.communication import ExecutorlibSocketError


def calc(i):
    return i


def resource_dict(resource_dict):
    return resource_dict


def get_worker_id(executorlib_worker_id):
    sleep(0.1)
    return executorlib_worker_id


def init_function():
    return {"a": 1, "b": 2}


def exit_funct():
    import sys
    sys.exit()


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial_with_dependencies(self):
        with SingleNodeExecutor(
            max_cores=2,
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
        with SingleNodeExecutor(
            max_cores=2,
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
        with SingleNodeExecutor(
            max_cores=1,
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
            SingleNodeExecutor(
                max_cores=1,
                resource_dict={
                    "cores": 1,
                    "gpus_per_core": 1,
                },
            )
        with self.assertRaises(ValueError):
            with SingleNodeExecutor(
                max_cores=1,
                block_allocation=False,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})
        with self.assertRaises(ValueError):
            with SingleNodeExecutor(
                max_cores=1,
                block_allocation=True,
            ) as exe:
                exe.submit(resource_dict, resource_dict={})


class TestWorkerID(unittest.TestCase):
    def test_block_allocation_True(self):
        with SingleNodeExecutor(
            max_cores=1,
            block_allocation=True,
        ) as exe:
            worker_id = exe.submit(get_worker_id, resource_dict={}).result()
        self.assertEqual(worker_id, 0)

    def test_block_allocation_True_two_workers(self):
        with SingleNodeExecutor(
            max_cores=2,
            block_allocation=True,
        ) as exe:
            f1_worker_id = exe.submit(get_worker_id, resource_dict={})
            f2_worker_id = exe.submit(get_worker_id, resource_dict={})
        self.assertEqual(sum([f1_worker_id.result(), f2_worker_id.result()]), 1)

    def test_init_function(self):
        with SingleNodeExecutor(
            max_cores=1,
            block_allocation=True,
            init_function=init_function,
        ) as exe:
            worker_id = exe.submit(get_worker_id, resource_dict={}).result()
        self.assertEqual(worker_id, 0)

    def test_init_function_two_workers(self):
        with SingleNodeExecutor(
            max_cores=2,
            block_allocation=True,
            init_function=init_function,
        ) as exe:
            f1_worker_id = exe.submit(get_worker_id, resource_dict={})
            f2_worker_id = exe.submit(get_worker_id, resource_dict={})
        self.assertEqual(sum([f1_worker_id.result(), f2_worker_id.result()]), 1)

    def test_block_allocation_False(self):
        with SingleNodeExecutor(
            max_cores=1,
            block_allocation=False,
        ) as exe:
            worker_id = exe.submit(get_worker_id, resource_dict={}).result()
        self.assertEqual(worker_id, 0)

    def test_block_allocation_False_two_workers(self):
        with SingleNodeExecutor(
            max_cores=2,
            block_allocation=False,
        ) as exe:
            f1_worker_id = exe.submit(get_worker_id, resource_dict={})
            f2_worker_id = exe.submit(get_worker_id, resource_dict={})
        self.assertEqual(sum([f1_worker_id.result(), f2_worker_id.result()]), 0)


class TestFunctionCrashes(unittest.TestCase):
    def test_single_node_executor(self):
        with self.assertRaises(ExecutorlibSocketError):
            with SingleNodeExecutor() as exe:
                f = exe.submit(exit_funct)
                print(f.result())

    def test_single_node_executor_block_allocation(self):
        with self.assertRaises(ExecutorlibSocketError):
            with SingleNodeExecutor(block_allocation=True) as exe:
                f = exe.submit(exit_funct)
                print(f.result())

    def test_single_node_executor_init_function(self):
        with self.assertRaises(ExecutorlibSocketError):
            with SingleNodeExecutor(init_function=exit_funct, block_allocation=True) as exe:
                f = exe.submit(sum, [1, 1])
                print(f.result())
