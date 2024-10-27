import sys

from executorlib.cache.backend import backend_execute_task_in_file

if __name__ == "__main__":
    backend_execute_task_in_file(file_name=sys.argv[1])
