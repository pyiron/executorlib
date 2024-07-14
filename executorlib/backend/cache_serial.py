import sys

from executorlib.cache.shared import execute_task_in_file

if __name__ == "__main__":
    execute_task_in_file(file_name=sys.argv[1])
