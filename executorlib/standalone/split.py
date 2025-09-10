from concurrent.futures import Future
from typing import Any, Optional


class SplitFuture(Future):
    def __init__(self, future: Future, selector: int | str):
        super().__init__()
        self._future = future
        self._selector = selector

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def running(self) -> bool:
        return self._future.running()

    def done(self) -> bool:
        return self._future.done()

    def add_done_callback(self, fn) -> None:
        return self._future.add_done_callback(fn=fn)

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout=timeout)[self._selector]

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout=timeout)

    def set_running_or_notify_cancel(self) -> bool:
        return self._future.set_running_or_notify_cancel()

    def set_result(self, result: Any) -> None:
        return self._future.set_result(result=result)

    def set_exception(self, exception: Optional[BaseException]) -> None:
        return self._future.set_exception(exception=exception)


def split_future(future: Future, n: int) -> list[SplitFuture]:
    return [SplitFuture(future=future, selector=i) for i in range(n)]


def get_item_from_future(future: Future, key: str) -> SplitFuture:
    return SplitFuture(future=future, selector=key)
