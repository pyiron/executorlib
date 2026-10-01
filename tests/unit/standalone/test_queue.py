from concurrent.futures import Future, CancelledError
from queue import Queue
from threading import Thread
import time
import unittest

from executorlib.standalone.queue import cancel_items_in_queue, put_front


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

    def test_put_front_orders_ahead_of_existing_items(self):
        q = Queue()
        q.put("back")
        put_front(q, "front")
        self.assertEqual(q.get(), "front")
        self.assertEqual(q.get(), "back")
        q.task_done()
        q.task_done()
        q.join()

    def test_put_front_wakes_consumer_blocked_on_empty_queue(self):
        # Mutating q.queue directly (q.queue.insert(0, item)) skips the not_empty
        # notification, so a thread already parked in q.get() never wakes up and the
        # item sits in the queue forever. put_front() must not have that problem.
        q = Queue()
        received = []

        def consume():
            received.append(q.get())

        consumer = Thread(target=consume)
        consumer.start()
        try:
            time.sleep(0.2)  # give the consumer time to block inside q.get()
            put_front(q, "woken")
            consumer.join(timeout=5)
            self.assertFalse(consumer.is_alive(), "consumer stayed blocked in get()")
            self.assertEqual(received, ["woken"])
        finally:
            consumer.join(timeout=5)
