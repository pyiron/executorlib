from unittest import TestCase
from concurrent.futures import Future
from executorlib.standalone.batched import batched_futures


class TestBatched(TestCase):
    def test_batched_futures(self):
        lst = []
        for i in list(range(10)):
            f = Future()
            f.set_result(i)
            lst.append(f)
        self.assertEqual(batched_futures(lst=lst, n=3, skip_lst=[]), [0, 1, 2])
        self.assertEqual(batched_futures(lst=lst, skip_lst=[[0, 1, 2]], n=3), [3, 4, 5])
        self.assertEqual(batched_futures(lst=lst, skip_lst=[[0, 1, 2], [3, 4, 5]], n=3), [6, 7, 8])
        self.assertEqual(batched_futures(lst=lst, skip_lst=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], n=3), [9])

    def test_batched_futures_not_finished(self):
        lst = []
        for _ in list(range(10)):
            f = Future()
            lst.append(f)
        self.assertEqual(batched_futures(lst=lst, n=3, skip_lst=[]), [])
