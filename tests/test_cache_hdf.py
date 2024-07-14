import os
import shutil
import unittest


try:
    from executorlib.cache.hdf import dump, load

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True


def my_funct(a, b):
    return a + b


@unittest.skipIf(
    skip_h5io_test, "h5io is not installed, so the h5io tests are skipped."
)
class TestSharedFunctions(unittest.TestCase):
    def test_hdf_mixed(self):
        cache_directory = os.path.abspath("cache")
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

    def test_hdf_args(self):
        cache_directory = os.path.abspath("cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_args.h5")
        a = 1
        b = 2
        dump(file_name=file_name, data_dict={"fn": my_funct, "args": [a, b]})
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], [a, b])
        self.assertEqual(data_dict["kwargs"], {})

    def test_hdf_kwargs(self):
        cache_directory = os.path.abspath("cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_kwargs.h5")
        a = 1
        b = 2
        dump(
            file_name=file_name,
            data_dict={"fn": my_funct, "args": (), "kwargs": {"a": a, "b": b}},
        )
        data_dict = load(file_name=file_name)
        self.assertTrue("fn" in data_dict.keys())
        self.assertEqual(data_dict["args"], ())
        self.assertEqual(data_dict["kwargs"], {"a": a, "b": b})

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")
