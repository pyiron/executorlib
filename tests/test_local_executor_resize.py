import unittest
from executorlib import SingleNodeExecutor
from executorlib.standalone.serialize import cloudpickle_register


def sleep_funct(sec):
    from time import sleep
    sleep(sec)
    return sec


class TestResizing(unittest.TestCase):
    def test_without_dependencies_decrease(self):
        cloudpickle_register(ind=1)
        with SingleNodeExecutor(max_workers=2, block_allocation=True, disable_dependencies=True) as exe:
            future_lst = [exe.submit(sleep_funct, 1) for _ in range(4)]
            self.assertEqual([f.done() for f in future_lst], [False, False, False, False])
            self.assertEqual(len(exe), 4)
            sleep_funct(sec=0.5)
            exe.max_workers = 1
            self.assertTrue(len(exe) >= 1)
            self.assertEqual(len(exe._process), 1)
            self.assertEqual([f.done() for f in future_lst], [True, True, False, False])
            self.assertEqual([f.result() for f in future_lst], [1, 1, 1, 1])
            self.assertEqual([f.done() for f in future_lst], [True, True, True, True])

    def test_without_dependencies_increase(self):
        cloudpickle_register(ind=1)
        with SingleNodeExecutor(max_workers=1, block_allocation=True, disable_dependencies=True) as exe:
            future_lst = [exe.submit(sleep_funct, 0.1) for _ in range(4)]
            self.assertEqual([f.done() for f in future_lst], [False, False, False, False])
            self.assertEqual(len(exe), 4)
            future_lst[0].result()
            exe.max_workers = 2
            self.assertEqual(len(exe), 2)
            self.assertEqual(len(exe._process), 2)
            self.assertEqual([f.done() for f in future_lst], [True, False, False, False])
            self.assertEqual([f.result() for f in future_lst], [0.1, 0.1, 0.1, 0.1])
            self.assertEqual([f.done() for f in future_lst], [True, True, True, True])