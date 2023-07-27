import unittest
from concurrent.futures import Future, CancelledError
from queue import Queue
from pympipool.shared.taskexecutor import cancel_items_in_queue


class TestQueue(unittest.TestCase):
    def test_cancel_items_in_queue(self):
        q = Queue()
        fs1 = Future()
        fs2 = Future()
        q.put({"future": fs1})
        q.put({"future": fs2})
        cancel_items_in_queue(que=q)
        self.assertEqual(q.qsize(), 0)
        self.assertTrue(fs1.done())
        with self.assertRaises(CancelledError):
            self.assertTrue(fs1.result())
        self.assertTrue(fs2.done())
        with self.assertRaises(CancelledError):
            self.assertTrue(fs2.result())
        q.join()
