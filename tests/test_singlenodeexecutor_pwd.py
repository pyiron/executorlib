import json
import unittest
from executorlib import SingleNodeExecutor, get_item_from_future


def get_sum(x, y):
    return x + y
    
def get_prod_and_div(x, y):
    return {"prod": x * y, "div": x / y}

def get_square(x):
    return x ** 2


class TestPythonWorkflowDefinition(unittest.TestCase):
    def test_arithmetic(self):
        with SingleNodeExecutor(export_workflow_filename="workflow.json") as exe:
            future_prod_and_div = exe.submit(get_prod_and_div, x=1, y=2)
            future_prod = get_item_from_future(future_prod_and_div, key="prod")
            future_div = get_item_from_future(future_prod_and_div, key="div")
            future_sum = exe.submit(get_sum, x=future_prod, y=future_div)
            future_result = exe.submit(get_square, x=future_sum)

        with open("workflow.json", "r") as f:
            content = json.load(f)

        self.assertEqual(len(content["nodes"]), 6)
        self.assertEqual(len(content["edges"]), 6)
