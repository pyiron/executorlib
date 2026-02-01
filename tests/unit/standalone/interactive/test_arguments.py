from concurrent.futures import Future
import unittest

from executorlib.standalone.interactive.arguments import (
    check_exception_was_raised,
    get_exception_lst,
    get_future_objects_from_input,
    update_futures_in_input,
)


class TestSerial(unittest.TestCase):
    def test_get_future_objects_from_input_with_future(self):
        input_args = (1, 2, Future(), [Future()], {3: Future()})
        input_kwargs = {"a": 1, "b": [Future()], "c": {"d": Future()}, "e": Future()}
        future_lst, boolean_flag = get_future_objects_from_input(args=input_args, kwargs=input_kwargs)
        self.assertEqual(len(future_lst), 6)
        self.assertFalse(boolean_flag)

    def test_get_future_objects_from_input_without_future(self):
        input_args = (1, 2)
        input_kwargs = {"a": 1}
        future_lst, boolean_flag = get_future_objects_from_input(args=input_args, kwargs=input_kwargs)
        self.assertEqual(len(future_lst), 0)
        self.assertTrue(boolean_flag)

    def test_update_futures_in_input_with_future(self):
        f1 = Future()
        f1.set_result(1)
        f2 = Future()
        f2.set_result(2)
        f3 = Future()
        f3.set_result(3)
        f4 = Future()
        f4.set_result(4)
        f5 = Future()
        f5.set_result(5)
        f6 = Future()
        f6.set_result(6)
        input_args = (1, 2, f1, [f2], {3: f3})
        input_kwargs = {"a": 1, "b": [f4], "c": {"d": f5}, "e": f6}
        output_args, output_kwargs = update_futures_in_input(args=input_args, kwargs=input_kwargs)
        self.assertEqual(output_args, (1, 2, 1, [2], {3: 3}))
        self.assertEqual(output_kwargs, {"a": 1, "b": [4], "c": {"d": 5}, "e": 6})

    def test_update_futures_in_input_without_future(self):
        input_args = (1, 2)
        input_kwargs = {"a": 1}
        output_args, output_kwargs = update_futures_in_input(args=input_args, kwargs=input_kwargs)
        self.assertEqual(input_args, output_args)
        self.assertEqual(input_kwargs, output_kwargs)

    def test_check_exception_was_raised(self):
        f_with_exception = Future()
        f_with_exception.set_exception(ValueError())
        f_without_exception = Future()
        self.assertTrue(check_exception_was_raised(future_obj=f_with_exception))
        self.assertFalse(check_exception_was_raised(future_obj=f_without_exception))

    def test_get_exception_lst(self):
        f_with_exception = Future()
        f_with_exception.set_exception(ValueError())
        f_without_exception = Future()
        future_with_exception_lst = [f_with_exception, f_with_exception, f_without_exception, f_without_exception, f_with_exception]
        future_without_exception_lst = [f_without_exception, f_without_exception, f_without_exception, f_without_exception]
        exception_lst = get_exception_lst(future_lst=future_with_exception_lst)
        self.assertEqual(len(exception_lst), 3)
        exception_lst = get_exception_lst(future_lst=future_without_exception_lst)
        self.assertEqual(len(exception_lst), 0)
