import unittest
from pympipool import Pool


def echo_funct(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


class TestTask(unittest.TestCase):
    def test_echo(self):
        with Pool(cores=2, enable_mpi4py_backend=False) as p:
            output = p.apply(echo_funct, 2)
        self.assertEqual(output, [2, 2])

    def test_mpi(self):
        with Pool(cores=2, enable_mpi4py_backend=False) as p:
            output = p.apply(mpi_funct, 2)
        self.assertEqual(output, [(2, 2, 0), (2, 2, 1)])

    def test_mpi_multiple(self):
        with Pool(cores=2, enable_mpi4py_backend=False) as p:
            p._interface.send_dict(input_dict={"f": mpi_funct, "a": [2]})
            p._interface.send_dict(input_dict={"f": mpi_funct, "a": [2]})
            p._interface.send_dict(input_dict={"f": mpi_funct, "a": [2]})
            output = [
                p._interface.receive_dict(),
                p._interface.receive_dict(),
                p._interface.receive_dict(),
            ]
        self.assertEqual(output, [
            [(2, 2, 0), (2, 2, 1)],
            [(2, 2, 0), (2, 2, 1)],
            [(2, 2, 0), (2, 2, 1)]
        ])
