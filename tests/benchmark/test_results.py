import unittest


class TestResults(unittest.TestCase):
    def test_result(self):
        with open("timing.log") as f:
            content = f.readlines()
        timing_dict = {l.split()[0]: float(l.split()[1]) for l in content}
        self.assertEqual(min(timing_dict, key=timing_dict.get), "process")
        self.assertEqual(max(timing_dict, key=timing_dict.get), "static")
        self.assertTrue(timing_dict["process"] < timing_dict["pympipool"])
        self.assertTrue(timing_dict["pympipool"] < timing_dict["process"] * 1.1)
        self.assertTrue(timing_dict["process"] < timing_dict["mpi4py"])
        self.assertTrue(timing_dict["pympipool"] < timing_dict["mpi4py"])
        self.assertTrue(timing_dict["mpi4py"] < timing_dict["process"] * 1.15)
        self.assertTrue(timing_dict["thread"] < timing_dict["static"])
        self.assertTrue(timing_dict["mpi4py"] < timing_dict["thread"])
