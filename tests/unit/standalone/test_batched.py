from unittest import TestCase
from concurrent.futures import Future
from executorlib.standalone.batched import batched_futures, _logged_failed_ids


class TestBatched(TestCase):
    def setUp(self):
        _logged_failed_ids.clear()

    def tearDown(self):
        _logged_failed_ids.clear()

    def test_batched_futures(self):
        lst = []
        for i in list(range(10)):
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

    def test_batched_futures_not_finished(self):
        lst = []
        for _ in list(range(10)):
            f = Future()
            lst.append(f)
        self.assertEqual(batched_futures(lst=lst, n=3, skip_lst=[]), [])

    def test_batched_futures_with_failed_future(self):
        """Failed futures are excluded from the batch rather than raising."""
        lst = []
        for i in range(5):
            f = Future()
            f.set_result(i)
            lst.append(f)
        f_failed = Future()
        f_failed.set_exception(RuntimeError("task failed"))
        lst.insert(2, f_failed)  # insert at position 2: [0, 1, FAILED, 2, 3, 4]
        # The failed future must not propagate; first 3 successful results are returned
        result = batched_futures(lst=lst, n=3, skip_lst=[])
        self.assertEqual(result, [0, 1, 2])
        # The failed future's id is recorded so it is only logged once
        self.assertIn(id(f_failed), _logged_failed_ids)

    def test_batched_futures_failed_future_logged_once(self):
        """A failed future is only logged once, even across multiple calls."""
        f_failed = Future()
        f_failed.set_exception(RuntimeError("task failed"))
        lst = [f_failed]
        batched_futures(lst=lst, n=1, skip_lst=[])
        self.assertIn(id(f_failed), _logged_failed_ids)
        size_after_first_call = len(_logged_failed_ids)
        # Second call must not add the id again
        batched_futures(lst=lst, n=1, skip_lst=[])
        self.assertEqual(len(_logged_failed_ids), size_after_first_call)

    def test_batched_futures_partial_batch_due_to_failures(self):
        """Emit a partial batch when all futures are resolved but n is unreachable due to failures."""
        lst = []
        for i in range(2):
            f = Future()
            f.set_result(i)
            lst.append(f)
        f_failed = Future()
        f_failed.set_exception(RuntimeError("task failed"))
        lst.append(f_failed)
        # all_resolved=True, only 2 successful results remain — must emit partial batch [0, 1]
        result = batched_futures(lst=lst, n=3, skip_lst=[])
        self.assertEqual(result, [0, 1])
