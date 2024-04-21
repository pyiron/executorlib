import sys
from time import time


def llh_numpy(mean, sigma):
    import numpy

    data = numpy.random.normal(size=100000000).astype("float64")
    s = (data - mean) ** 2 / (2 * (sigma**2))
    pdfs = numpy.exp(-s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()


def run_with_executor(executor=None, mean=0.1, sigma=1.1, runs=32, **kwargs):
    with executor(**kwargs) as exe:
        future_lst = [
            exe.submit(llh_numpy, mean=mean, sigma=sigma) for i in range(runs)
        ]
        return [f.result() for f in future_lst]


def run_static(mean=0.1, sigma=1.1, runs=32):
    return [llh_numpy(mean=mean, sigma=sigma) for i in range(runs)]


if __name__ == "__main__":
    run_mode = sys.argv[1]
    start_time = time()
    if run_mode == "static":
        run_static(mean=0.1, sigma=1.1, runs=32)
    elif run_mode == "process":
        from concurrent.futures import ProcessPoolExecutor

        run_with_executor(
            executor=ProcessPoolExecutor, mean=0.1, sigma=1.1, runs=32, max_workers=4
        )
    elif run_mode == "thread":
        from concurrent.futures import ThreadPoolExecutor

        run_with_executor(
            executor=ThreadPoolExecutor, mean=0.1, sigma=1.1, runs=32, max_workers=4
        )
    elif run_mode == "pympipool":
        from pympipool import Executor

        run_with_executor(
            executor=Executor, mean=0.1, sigma=1.1, runs=32, max_cores=4, backend="mpi"
        )
    elif run_mode == "flux":
        from pympipool import Executor

        run_with_executor(
            executor=Executor, mean=0.1, sigma=1.1, runs=32, max_cores=4, backend="flux"
        )
    elif run_mode == "mpi4py":
        from mpi4py.futures import MPIPoolExecutor

        run_with_executor(
            executor=MPIPoolExecutor, mean=0.1, sigma=1.1, runs=32, max_workers=4
        )
    else:
        raise ValueError(run_mode)
    stop_time = time()
    print(run_mode, stop_time - start_time)
