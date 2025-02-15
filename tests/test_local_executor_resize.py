import unittest
from executorlib import SingleNodeExecutor


def sleep_funct(sec):
    from time import sleep
    sleep(sec)
    return sec


class TestResizing(unittest.TestCase):
    def test_without_dependencies(self):
        with SingleNodeExecutor(max_workers=2, block_allocation=True, disable_dependencies=True) as exe:
            future_lst = [exe.submit(sleep_funct, 0.1) for _ in range(4)]
            self.assertEqual([f.done() for f in future_lst], [False, False, False, False])
            self.assertEqual(len(exe), 4)
            sleep_funct(sec=0.01)
            exe.max_workers = 1
            self.assertEqual(len(exe), 1)
            self.assertEqual(len(exe._process), 1)
            self.assertEqual([f.done() for f in future_lst], [True, True, False, False])
            self.assertEqual([f.result() for f in future_lst], [0.1, 0.1, 0.1, 0.1])
