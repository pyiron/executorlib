try:
    from mpi4py.MPI import Request, SUM, MAX, MIN, IN_PLACE, IDENT, CONGRUENT, SIMILAR, UNEQUAL
except ImportError:
    pass

import numpy as np


class MPI4PYWrapper:
    def __init__(self, comm, parent=None):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.parent = parent  # XXX check C-object against comm.parent?

    def new_communicator(self, ranks):
        comm = self.comm.Create(self.comm.group.Incl(ranks))
        if self.comm.rank in ranks:
            return MPI4PYWrapper(comm, parent=self)
        else:
            # This cpu is not in the new communicator:
            return None

    def max_scalar(self, a, root=-1):
        return self.sum_scalar(a, root=-1, _op=MAX)

    def min_scalar(self, a, root=-1):
        return self.sum_scalar(a, root=-1, _op=MIN)

    def sum_scalar(self, a, root=-1, _op=None):
        if _op is None:
            _op = SUM
        assert isinstance(a, (int, float, complex))
        if root == -1:
            return self.comm.allreduce(a, op=_op)
        else:
            return self.comm.reduce(a, root=root, op=_op)

    def sum(self, a, root=-1):
        if root == -1:
            self.comm.Allreduce(IN_PLACE, a, op=SUM)
        else:
            if root == self.rank:
                self.comm.Reduce(IN_PLACE, a, root=root, op=SUM)
            else:
                self.comm.Reduce(a, None, root=root, op=SUM)

    def scatter(self, a, b, root):
        self.comm.Scatter(a, b, root)

    def alltoallv(self, sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls):
        self.comm.Alltoallv((sbuffer, (scounts, sdispls), sbuffer.dtype.char),
                            (rbuffer, (rcounts, rdispls), rbuffer.dtype.char))

    def all_gather(self, a, b):
        self.comm.Allgather(a, b)

    def gather(self, a, root, b=None):
        self.comm.Gather(a, b, root)

    def broadcast(self, a, root):
        self.comm.Bcast(a, root)

    def sendreceive(self, a, dest, b, src, sendtag=123, recvtag=123):
        return self.comm.Sendrecv(a, dest, sendtag, b, src, recvtag)

    def send(self, a, dest, tag=123, block=True):
        if block:
            self.comm.Send(a, dest, tag)
        else:
            return self.comm.Isend(a, dest, tag)

    def ssend(self, a, dest, tag=123):
        return self.comm.Ssend(a, dest, tag)

    def receive(self, a, src, tag=123, block=True):
        if block:
            self.comm.Recv(a, src, tag)
        else:
            return self.comm.Irecv(a, src, tag)

    def test(self, request):
        return request.test()

    def testall(self, requests):
        return Request.testall(requests)

    def wait(self, request):
        request.wait()

    def waitall(self, requests):
        Request.waitall(requests)

    def name(self):
        return self.comm.Get_name()

    def barrier(self):
        self.comm.barrier()

    def abort(self, errcode):
        self.comm.Abort(errcode)

    def compare(self, othercomm):
        code = self.comm.Compare(othercomm.comm)
        if code == IDENT:
            return "ident"
        elif code == CONGRUENT:
            return "congruent"
        elif code == SIMILAR:
            return "similar"
        elif code == UNEQUAL:
            return "unequal"
        else:
            raise ValueError(f"Unknown compare code {code}")

    def translate_ranks(self, other, ranks):
        return np.array(self.comm.Get_group().Translate_ranks(ranks, other.comm.Get_group()))

    def get_members(self):
        return self.translate_ranks(self.parent, np.arange(self.size))

    def get_c_object(self):
        return self.comm
