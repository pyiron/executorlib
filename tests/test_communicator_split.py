import unittest
from pympipool import Pool


def get_ranks(input_parameter, comm=None):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if comm is not None:
        size_new = comm.Get_size()
        rank_new = comm.Get_rank()
    else:
        size_new = 0
        rank_new = 0
    return size, rank, size_new, rank_new, input_parameter


class TestPool(unittest.TestCase):
    def test_pool_serial(self):
        with Pool(cores=4, cores_per_task=1) as p:
            output = p.map(function=get_ranks, lst=[1, 2, 3, 4])
        self.assertEqual(output[0], [4, 0, 0, 0, 1])
        self.assertEqual(output[1], [4, 1, 0, 0, 2])
        self.assertEqual(output[2], [4, 2, 0, 0, 3])
        self.assertEqual(output[3], [4, 3, 0, 0, 4])

    def test_pool_parallel(self):
        with Pool(cores=4, cores_per_task=2) as p:
            output = p.map(function=get_ranks, lst=[1, 2, 3, 4])
        self.assertEqual(output[0], [4, 0, 2, 0, 1])
        self.assertEqual(output[1], [4, 1, 2, 0, 2])
        self.assertEqual(output[2], [4, 2, 2, 0, 3])
        self.assertEqual(output[3], [4, 3, 2, 0, 4])
