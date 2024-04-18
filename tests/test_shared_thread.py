import unittest
from pympipool.shared.thread import RaisingThread


def raise_error():
    raise ValueError


class TestRaisingThread(unittest.TestCase):
    def test_raising_thread(self):
        with self.assertRaises(ValueError):
            process = RaisingThread(target=raise_error)
            process.start()
            process.join()
