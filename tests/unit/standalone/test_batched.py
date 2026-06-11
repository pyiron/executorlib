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
        batched_lst[0].set_result([0, 1, 2])
        batched_lst[1].set_result([3, 4, 5])
        batched_lst[2].set_result([6, 7, 8])
        self.assertEqual(batched_futures(lst=lst, n=3, nested_skip_lst=set()), [0, 1, 2])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3), [3, 4, 5])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst[:2], n=3), [6, 7, 8])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst, n=3), [9])

    def test_batched_futures_duplicated(self):
        lst = []
        for i in range(1,4):
            for _ in range(3):
                f = Future()
                f.set_result(i)
                lst.append(f)
        batched_lst = [Future(), Future(), Future()]
        batched_lst[0].set_result([1, 1, 1])
        batched_lst[1].set_result([2, 2, 2])
        batched_lst[2].set_result([3, 3, 3])
        self.assertEqual(batched_futures(lst=lst, n=3, nested_skip_lst=set()), [1, 1, 1])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3), [2, 2, 2])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst[:2], n=3), [3, 3, 3])

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
        batched_lst[0].set_result([1, 2, 4])
        batched_lst[1].set_result([5, 7, 8])
        self.assertEqual(batched_futures(lst=lst, n=3, nested_skip_lst=set()), [1, 2, 4])
        self.assertEqual(batched_futures(lst=lst, nested_skip_lst=batched_lst[:1], n=3), [5, 7, 8])
        with self.assertRaises(ValueError):
            batched_futures(lst=lst, nested_skip_lst=batched_lst, n=3)

    def test_batched_futures_not_finished(self):
        lst = []
        for _ in list(range(10)):
            f = Future()
            lst.append(f)
        self.assertEqual(batched_futures(lst=lst, n=3, nested_skip_lst=set()), [])
