from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Define your G-Eval metrics
explanation_type_metric = GEval(
    name="Explanation Type",
    evaluation_steps=[
        """1. Given the below 5 numbered categories of explanation, assign a score matching the most advanced explanation type present in the answer.
{
    "explanation_types": {
        {
            "score": 0,
            "type": "Absent",
            "description": "No explanation provided.",
            "example": ""
        },
        {
            "score": 2.5,
            "type": "Definition",
            "description": "A short definition of a certain entity is present, without further explanation.",
            "example": "The internet is a virtual network."
        },
        {
            "score": 5,
            "type": "Elucidating",
            "description": "A definition with an example.",
            "example": "Antibiotics only work on bacteria, which means that they can only be used for diseases caused by microbes belonging to the bacteria family."
        },
        {
            "score": 7.5,
            "type": "Transformative",
            "description": "Any explanation whose starting point is what the audience might think, that points to problems with the existing conceptions.",
            "example": "I believe that the Bible must be interpreted in the context in which it was written..."
        },{
            "score": 10,
            "type": "Quasiscientific",
            "description": "An explanation that creates an image in the mind, often by using an analogy.",
            "example": "Consider each computer as a node and the Internet as a web."
        }
    }
}""",
        "2. When scoring, do not consider correctness. Instead, follow the descriptions in step 1 to determine the score.",
        "3. If an answer contains multiple types of explanations, assign the score based on the best explanation type in the answer.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

content_units_metric = GEval(
    name="Content Units",
    evaluation_steps=[
        '1. A standalone fact is a fact that does not depend on other facts. Identify and extract all standalone facts from the Actual Output.',
        '2. Count each standalone fact as a separate content unit.',
        '3. Pay no attention to other dimensions such as factual correctness.',
        '4. Return the amount of content units present in the Actual Output.',
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

connection_to_everyday_life_metric = GEval(
    name="Connection to everyday life",
    evaluation_steps=[
        """1.Check the output contains an explicit connection to common knowledge, a previous event, or a news story that was not already embedded in the question.""",
        "2. Return a score of 10 if the above holds, and a score of 0 otherwise."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

humor_metric = GEval(
    name="Humor",
    evaluation_steps=[
        "1. Determine if the explanation includes explicit jokes or ironic language.",
        "2. Return a score of 10 if jokes or ironic language are present in the answer, and 0 otherwise.",
        "3. If you aren't sure whether the answer contains jokes or ironic language, return a score of 5."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

analogy_metric = GEval(
    name="Analogy",
    evaluation_steps=[
        """1. Consider the following definition of analogies: Analogies are defined as a systematic mapping between two situations:
the source (familiar situation) and the target (novel situation).
        """,
        "2. Based on the above definition, determine whether the explanation includes analogies or not. Do not take correctness into account.",
        "3. Return a score of 10 if at least one analogy is present in the answer, and 0 if no analogies are present in the answer.",
        "4. If you aren't sure whether the answer contains a analogy or not, return a score of 5.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

metaphor_metric = GEval(
    name="Metaphor",
    evaluation_steps=[
        """1. Consider the following definition of metaphors: Metaphors structure one concept in terms of another. Unlike
analogies, metaphors do not necessarily map directly between source and
target; similarities can be associative.
        """,
        "2. Based on the above definition, determine whether the explanation includes metaphors or not. Do not take correctness into account.",
        "3. Return a score of 10 if at least one metaphor is present in the answer, and 0 if no metaphors are present in the answer.",
        "4. If you aren't sure whether the answer contains a metaphor or not, return a score of 5.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "1. Determine whether the actual output is factually correct based on the expected output.",
        "2. Return a grade on a scale from 0 to 10 where 0 is completely false, and 10 is completely true.",
    ],
    evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)