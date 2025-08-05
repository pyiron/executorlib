import os
import shutil
import unittest


try:
    from executorlib.standalone.hdf import (
        dump_to_hdf,
        load_from_hdf,
        get_output_from_hdf,
        get_runtime_from_hdf,
        get_queue_id_from_hdf,
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
        dump_to_hdf(
            file_name=file_name,
            data_dict={"fn": my_funct, "args": [a], "kwargs": {"b": b}},
        )
        data_dict = load_from_hdf(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], [a])
        self.assertEqual(data_dict["kwargs"], {"b": b})
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertFalse(no_error)
        self.assertFalse(flag)
        self.assertIsNone(output)

    def test_hdf_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_args.h5")
        a = 1
        b = 2
        dump_to_hdf(file_name=file_name, data_dict={"fn": my_funct, "args": [a, b]})
        data_dict = load_from_hdf(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], [a, b])
        self.assertEqual(data_dict["kwargs"], {})
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_kwargs(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_kwargs.h5")
        a = 1
        b = 2
        dump_to_hdf(
            file_name=file_name,
            data_dict={
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": a, "b": b},
                "queue_id": 123,
                "error_log_file": "error.out",
            },
        )
        data_dict = load_from_hdf(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], ())
        self.assertEqual(data_dict["kwargs"], {"a": a, "b": b})
        self.assertEqual(get_queue_id_from_hdf(file_name=file_name), 123)
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_missing_funct(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_missing_funct.h5")
        dump_to_hdf(
            file_name=file_name,
            data_dict={
                "queue_id": 123,
            },
        )
        with self.assertRaises(TypeError):
            load_from_hdf(file_name=file_name)

    def test_hdf_missing_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_missing_args.h5")
        dump_to_hdf(
            file_name=file_name,
            data_dict={
                "fn": my_funct,
            },
        )
        data_dict = load_from_hdf(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], ())
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_queue_id(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_queue.h5")
        queue_id = 123
        dump_to_hdf(
            file_name=file_name,
            data_dict={"queue_id": queue_id},
        )
        self.assertEqual(get_queue_id_from_hdf(file_name=file_name), 123)
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertFalse(flag)
        self.assertFalse(no_error)
        self.assertIsNone(output)

    def test_hdf_error(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_error.h5")
        error = ValueError()
        dump_to_hdf(
            file_name=file_name,
            data_dict={"error": error},
        )
        flag, no_error, output = get_output_from_hdf(file_name=file_name)
        self.assertTrue(get_runtime_from_hdf(file_name=file_name) == 0.0)
        self.assertTrue(flag)
        self.assertFalse(no_error)
        self.assertTrue(isinstance(output, error.__class__))

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)
