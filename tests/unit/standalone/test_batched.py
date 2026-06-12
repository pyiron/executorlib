import unittest
from concurrent.futures import Future

from executorlib.task_scheduler.interactive.dependency import batched_futures


class TestBatched(unittest.TestCase):
    def test_batched_futures(self):
        lst = []
        for i in range(10):
            f = Future()
            f.set_result(i)
            lst.append(f)
        batched_lst = [Future(), Future(), Future()]
        batched_lst[0].set_result([id(lst[0]), id(lst[1]), id(lst[2])])
        batched_lst[1].set_result([id(lst[3]), id(lst[4]), id(lst[5])])
        batched_lst[2].set_result([id(lst[6]), id(lst[7]), id(lst[8])])
        success, done_lst = batched_futures(lst=lst, n=3, nested_skip_lst=set())
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [0, 1, 2])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [3, 4, 5])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst[:2], n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [6, 7, 8])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst, n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [9])

    def test_batched_futures_duplicated(self):
        lst = []
        for i in range(1,4):
            for _ in range(3):
                f = Future()
                f.set_result(i)
                lst.append(f)
        batched_lst = [Future(), Future(), Future()]
        batched_lst[0].set_result([id(lst[0]), id(lst[1]), id(lst[2])])
        batched_lst[1].set_result([id(lst[3]), id(lst[4]), id(lst[5])])
        batched_lst[2].set_result([id(lst[6]), id(lst[7]), id(lst[8])])
        success, done_lst = batched_futures(lst=lst, n=3, nested_skip_lst=set())
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [1, 1, 1])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [2, 2, 2])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst[:2], n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [3, 3, 3])

    def test_batched_futures(self):
        lst = []
        for i in range(10):
            f = Future()
            if i % 3 == 0:
                f.set_exception(ValueError(f"Error for {i}"))
            else:
                f.set_result(i)
            lst.append(f)
        batched_lst = [Future(), Future()]
        batched_lst[0].set_result([id(lst[1]), id(lst[2]), id(lst[4])])
        batched_lst[1].set_result([id(lst[5]), id(lst[7]), id(lst[8])])
        success, done_lst = batched_futures(lst=lst, n=3, nested_skip_lst=set())
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [1, 2, 4])
        success, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3)
        self.assertTrue(success)
        self.assertEqual([f.result() for f in done_lst], [5, 7, 8])
        succss, done_lst = batched_futures(lst=lst, nested_skip_lst=batched_lst, n=3)
        self.assertFalse(succss)
        with self.assertRaises(ValueError):
            raise done_lst[0].exception()

    def test_batched_futures_not_finished(self):
        lst = []
        for _ in list(range(10)):
            f = Future()
            lst.append(f)
        success, done_lst = batched_futures(lst=lst, n=3, nested_skip_lst=set())
        self.assertTrue(success)
        self.assertEqual(done_lst, [])
