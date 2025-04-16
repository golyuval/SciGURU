from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

content_units_metric = GEval(
    name="Content Units",
    evaluation_steps=[
        '1. A standalone fact is a fact that does not depend on other facts. Identify and extract all standalone facts from the Actual Output.',
        '2. Count each standalone fact as a separate content unit.',
        '3. Pay no attention to other dimensions such as factual correctness.',
        '4. Return the amount of content units present in the Actual Output.',
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

connection_to_everyday_life_metric = GEval(
    name="Connection to everyday life",
    evaluation_steps=[
        """1. Check the output contains an explicit connection to common knowledge, a previous event, or a news
story that was not already embedded in the question.""",
        "2. Return a score of 10 if the above holds, and a score of 0 otherwise."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)