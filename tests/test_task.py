import unittest
from pympipool.mpi.executor import PyMPISingleTaskExecutor


def echo_funct(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


class TestTask(unittest.TestCase):
    def test_echo(self):
        with PyMPISingleTaskExecutor(cores=2, hostname_localhost=True) as p:
            output = p.submit(echo_funct, 2).result()
        self.assertEqual(output, [2, 2])

    def test_mpi(self):
        with PyMPISingleTaskExecutor(cores=2, hostname_localhost=True) as p:
            output = p.submit(mpi_funct, 2).result()
        self.assertEqual(output, [(2, 2, 0), (2, 2, 1)])

    def test_mpi_multiple(self):
        with PyMPISingleTaskExecutor(cores=2, hostname_localhost=True) as p:
            fs1 = p.submit(mpi_funct, 1)
            fs2 = p.submit(mpi_funct, 2)
            fs3 = p.submit(mpi_funct, 3)
            output = [
                fs1.result(),
                fs2.result(),
                fs3.result(),
            ]
        self.assertEqual(
            output,
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )
