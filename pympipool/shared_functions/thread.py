from threading import Thread


class RaisingThread(Thread):
    """
    Based on https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
    """

    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ):
        super().__init__(
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )
        self._exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self._exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self._exception:
            raise self._exception
