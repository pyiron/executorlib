from pympipool.scheduler import create_executor
from pympipool.shared.executorbase import ExecutorSteps, execute_tasks_with_dependencies
from pympipool.shared.thread import RaisingThread


class ExecutorWithDependencies(ExecutorSteps):
    def __init__(self, *args, refresh_rate: float = 0.01, **kwargs):
        super().__init__()
        executor = create_executor(*args, **kwargs)
        self._set_process(
            RaisingThread(
                target=execute_tasks_with_dependencies,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "executor_queue": executor._future_queue,
                    "executor": executor,
                    "refresh_rate": refresh_rate,
                },
            )
        )
