from concurrent.futures import Future
import os
import shutil
import unittest

from executorlib.standalone.select import FutureSelector


try:
    from executorlib.task_scheduler.file.backend import backend_execute_task_in_file
    from executorlib.task_scheduler.file.shared import _check_task_output, _convert_args_and_kwargs, FutureItem
    from executorlib.standalone.hdf import dump, get_runtime
    from executorlib.standalone.serialize import serialize_funct

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True


def my_funct(a, b):
    return a + b


def return_dict(a, b):
    return {"a": a, "b": b}


def return_list(a, b):
    return [a, b]


def get_error(a):
    raise ValueError(a)


@unittest.skipIf(
    skip_h5io_test, "h5io is not installed, so the h5io tests are skipped."
)
class TestSharedFunctions(unittest.TestCase):
    def test_execute_function_mixed(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_mixed_selector_convert(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key_1, data_dict = serialize_funct(
            fn=return_dict,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name_1 = os.path.join(cache_directory, task_key_1 + "_i.h5")
        dump(file_name=file_name_1, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name_1)
        f1 = Future()
        _check_task_output(
            task_key=task_key_1, future_obj=f1, cache_directory=cache_directory
        )
        task_key_2, data_dict = serialize_funct(
            fn=return_list,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name_2 = os.path.join(cache_directory, task_key_2 + "_i.h5")
        dump(file_name=file_name_2, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name_2)
        f2 = Future()
        _check_task_output(
            task_key=task_key_2, future_obj=f2, cache_directory=cache_directory
        )
        fs1 = FutureSelector(future=f1, selector="a")
        fs2 = FutureSelector(future=f2, selector=0)
        task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
            task_dict={"fn": 1, "args": (fs1,), "kwargs": {"a": fs2}},
            memory_dict={task_key_1: f1, task_key_2: f2},
            file_name_dict={
                task_key_1: os.path.join(cache_directory, task_key_1 + "_i.h5"), 
                task_key_2: os.path.join(cache_directory, task_key_2 + "_i.h5"),
            },
        )
        self.assertEqual(task_args[0].result(), 1)
        self.assertEqual(task_kwargs["a"].result(), 2)
        self.assertTrue(len(future_wait_key_lst) == 2)

    def test_execute_function_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=[1, 2],
            fn_kwargs=None,
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        os.makedirs(os.path.join(cache_directory, task_key), exist_ok=True)
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_kwargs(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=None,
            fn_kwargs={"a": 1, "b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_error(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=get_error,
            fn_args=[],
            fn_kwargs={"a": 1},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        data_dict["error_log_file"] = os.path.join(cache_directory, "error.out")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory
        )
        self.assertTrue(future_obj.done())
        with self.assertRaises(ValueError):
            future_obj.result()
        with open(os.path.join(cache_directory, "error.out"), "r") as f:
            content = f.readlines()
        self.assertEqual(content[1], 'args: []\n')
        self.assertEqual(content[2], "kwargs: {'a': 1}\n")
        self.assertEqual(content[-1], 'ValueError: 1\n')
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        with self.assertRaises(ValueError):
            future_file_obj.result()

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)
