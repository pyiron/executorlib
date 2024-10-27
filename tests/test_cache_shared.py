from concurrent.futures import Future
import os
import shutil
import unittest


try:
    from executorlib.cache.backend import backend_execute_task_in_file
    from executorlib.cache.shared import _check_task_output, FutureItem
    from executorlib.standalone.hdf import dump
    from executorlib.standalone.serialize import serialize_funct_h5

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True


def my_funct(a, b):
    return a + b


@unittest.skipIf(
    skip_h5io_test, "h5io is not installed, so the h5io tests are skipped."
)
class TestSharedFunctions(unittest.TestCase):
    def test_execute_function_mixed(self):
        cache_directory = os.path.abspath("cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct_h5(
            fn=my_funct,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + ".h5in")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + ".h5out")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_args(self):
        cache_directory = os.path.abspath("cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct_h5(
            fn=my_funct,
            fn_args=[1, 2],
            fn_kwargs={},
        )
        file_name = os.path.join(cache_directory, task_key + ".h5in")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + ".h5out")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_kwargs(self):
        cache_directory = os.path.abspath("cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct_h5(
            fn=my_funct,
            fn_args=[],
            fn_kwargs={"a": 1, "b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + ".h5in")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + ".h5out")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")
