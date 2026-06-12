import contextlib
import os
import shutil
import unittest
from concurrent.futures import Future
from typing import Optional

import numpy as np


try:
    from executorlib.standalone.hdf import (
        load,
        get_output,
        get_runtime,
        get_queue_id,
        get_future_from_cache,
        group_dict,
    )

    import h5py
    import cloudpickle

    def dump(file_name: Optional[str], data_dict: dict) -> None:
        """
        Previous dump function just copied here for backwards compatibility. 
        """
        if file_name is not None:
            file_name_abs = os.path.abspath(file_name)
            os.makedirs(os.path.dirname(file_name_abs), exist_ok=True)
            with h5py.File(file_name_abs, "a") as fname:
                for data_key, data_value in data_dict.items():
                    if data_key in group_dict:
                        with contextlib.suppress(ValueError):
                            fname.create_dataset(
                                name="/" + group_dict[data_key],
                                data=np.void(cloudpickle.dumps(data_value)),
                            )

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def my_funct(a, b):
    return a + b


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5io tests are skipped."
)
class TestSharedFunctions(unittest.TestCase):
    def test_hdf_mixed(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_mixed.h5")
        a = 1
        b = 2
        dump(
            file_name=file_name,
            data_dict={"fn": my_funct, "args": [a], "kwargs": {"b": b}},
        )
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], [a])
        self.assertEqual(data_dict["kwargs"], {"b": b})
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertFalse(no_error)
        self.assertFalse(flag)
        self.assertIsNone(output)

    def test_get_future_from_file(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_mixed_i.h5")
        a = 1
        b = 2
        dump(
            file_name=file_name,
            data_dict={"fn": my_funct, "args": [a], "kwargs": {"b": b}},
        )
        future = get_future_from_cache(
            cache_directory=cache_directory,
            cache_key="test_mixed",
        )
        self.assertTrue(isinstance(future, Future))
        self.assertFalse(future.done())

    def test_get_output_file_missing(self):
        cache_directory = os.path.abspath("executorlib_cache")
        with self.assertRaises(FileNotFoundError):
            get_output(file_name=os.path.join(cache_directory, "does_not_exist.h5"))

    def test_get_future_from_file_missing(self):
        cache_directory = os.path.abspath("executorlib_cache")
        with self.assertRaises(FileNotFoundError):
            get_future_from_cache(
            cache_directory=cache_directory,
            cache_key="does_not_exist",
        )

    def test_hdf_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_args.h5")
        a = 1
        b = 2
        dump(file_name=file_name, data_dict={"fn": my_funct, "args": [a, b]})
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], [a, b])
        self.assertEqual(data_dict["kwargs"], {})
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_kwargs(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_kwargs.h5")
        a = 1
        b = 2
        dump(
            file_name=file_name,
            data_dict={
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": a, "b": b},
                "queue_id": 123,
                "error_log_file": "error.out",
            },
        )
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], ())
        self.assertEqual(data_dict["kwargs"], {"a": a, "b": b})
        self.assertEqual(get_queue_id(file_name=file_name), 123)
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_missing_funct(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_missing_funct.h5")
        dump(
            file_name=file_name,
            data_dict={
                "queue_id": 123,
            },
        )
        with self.assertRaises(TypeError):
            load(file_name=file_name)

    def test_hdf_missing_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_missing_args.h5")
        dump(
            file_name=file_name,
            data_dict={
                "fn": my_funct,
            },
        )
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], ())
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_queue_id(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_queue.h5")
        queue_id = 123
        dump(
            file_name=file_name,
            data_dict={"queue_id": queue_id},
        )
        self.assertEqual(get_queue_id(file_name=file_name), 123)
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_error(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_error.h5")
        error = ValueError()
        dump(
            file_name=file_name,
            data_dict={"error": error},
        )
        flag, no_error, output = get_output(file_name=file_name)
        self.assertTrue(get_runtime(file_name=file_name) == 0.0)
        self.assertTrue(flag)
        self.assertFalse(no_error)
        self.assertTrue(isinstance(output, error.__class__))

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)
