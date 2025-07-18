import os
import shutil
import unittest

from executorlib import SingleNodeExecutor, get_cache_data
from executorlib.standalone.serialize import cloudpickle_register

try:
    import h5py

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def get_error(a):
    raise ValueError(a)


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5io tests are skipped."
)
class TestCacheFunctions(unittest.TestCase):
    def test_cache_data(self):
        cache_directory = os.path.abspath("executorlib_cache")
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            self.assertTrue(exe)
            future_lst = [exe.submit(sum, [i, i]) for i in range(1, 4)]
            result_lst = [f.result() for f in future_lst]

        cache_lst = get_cache_data(cache_directory=cache_directory)
        self.assertEqual(sum([c["output"] for c in cache_lst]), sum(result_lst))
        self.assertEqual(
            sum([sum(c["input_args"][0]) for c in cache_lst]), sum(result_lst)
        )

    def test_cache_key(self):
        cache_directory = os.path.abspath("executorlib_cache")
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            self.assertTrue(exe)
            future_lst = [exe.submit(sum, [i, i], resource_dict={"cache_key": "same/j" + str(i)}) for i in range(1, 4)]
            result_lst = [f.result() for f in future_lst]

        cache_lst = get_cache_data(cache_directory=cache_directory)
        for entry in cache_lst:
            self.assertTrue("same" in entry['filename'])
        self.assertEqual(sum([c["output"] for c in cache_lst]), sum(result_lst))
        self.assertEqual(
            sum([sum(c["input_args"][0]) for c in cache_lst]), sum(result_lst)
        )

    def test_cache_error(self):
        cache_directory = os.path.abspath("cache_error")
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            self.assertTrue(exe)
            cloudpickle_register(ind=1)
            f = exe.submit(get_error, a=1)
            with self.assertRaises(ValueError):
                print(f.result())

    def test_cache_error_file(self):
        cache_directory = os.path.abspath("cache_error")
        error_out =  "error.out"
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            self.assertTrue(exe)
            cloudpickle_register(ind=1)
            f = exe.submit(get_error, a=1, resource_dict={"error_log_file": error_out})
            with self.assertRaises(ValueError):
                print(f.result())
        self.assertTrue(os.path.exists(error_out))
        os.remove(error_out)

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)
        shutil.rmtree("cache_error", ignore_errors=True)
