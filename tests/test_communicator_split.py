import unittest
from pympipool import MPISpawnPool


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


def get_ranks_multi_input(input_parameter1, input_parameter2, comm=None):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if comm is not None:
        size_new = comm.Get_size()
        rank_new = comm.Get_rank()
    else:
        size_new = 0
        rank_new = 0
    return size, rank, size_new, rank_new, input_parameter1, input_parameter2


class TestCommunicator(unittest.TestCase):
    def test_map_serial(self):
        with MPISpawnPool(max_ranks=2, ranks_per_task=1) as p:
            output = p.map(func=get_ranks, iterable=[1, 2, 3])
        self.assertEqual(output[0], (2, 1, 0, 0, 1))
        self.assertEqual(output[1], (2, 1, 0, 0, 2))
        self.assertEqual(output[2], (2, 1, 0, 0, 3))

    def test_map_parallel(self):
        with MPISpawnPool(max_ranks=2, ranks_per_task=2) as p:
            output = p.map(func=get_ranks, iterable=[1, 2, 3, 4])
        self.assertEqual(output[0][::2], (2, 2, 1))
        self.assertEqual(output[1][::2], (2, 2, 2))
        self.assertEqual(output[2][::2], (2, 2, 3))
        self.assertEqual(output[3][::2], (2, 2, 4))

    def test_starmap_serial(self):
        with MPISpawnPool(max_ranks=2, ranks_per_task=1) as p:
            output = p.starmap(
                func=get_ranks_multi_input,
                iterable=[[1, 1], [2, 2], [3, 3]]
            )
        self.assertEqual(output[0], (2, 1, 0, 0, 1, 1))
        self.assertEqual(output[1], (2, 1, 0, 0, 2, 2))
        self.assertEqual(output[2], (2, 1, 0, 0, 3, 3))

    def test_starmap_parallel(self):
        with MPISpawnPool(max_ranks=2, ranks_per_task=2) as p:
            output = p.starmap(
                func=get_ranks_multi_input,
                iterable=[[1, 1], [2, 2], [3, 3], [4, 4]]
            )
        self.assertEqual(output[0][::2], (2, 2, 1))
        self.assertEqual(output[1][::2], (2, 2, 2))
        self.assertEqual(output[2][::2], (2, 2, 3))
        self.assertEqual(output[3][::2], (2, 2, 4))
