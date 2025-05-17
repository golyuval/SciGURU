from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


internal_coherence_metric_explicit = GEval(
    name="Internal Coherence Explicit",
    evaluation_steps=["1. Determine whether the parts of this explanation fit together coherently.",
                      "2. Output a grade between 0 and 10 based on the above criterion.",
                      "3. A grade of 10 means the parts of the explanation fit together perfectly coherently and 0 means that they don't fit at all.",
                      "4. Do not take any other factors into account when giving a grade - correctness or accuracy should not affect the grade."
                      ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

completeness_metric_explicit = GEval(
    name="Completeness (Explicit)",
    evaluation_steps=[
        "1. Read and understand the following definition: if an explanation suggests that A causes B, but it is not immediately clear how A causes B, that is considered a gap.",
        "2. Given the input question and output model answer, determine if the answer has gaps or not.",
        "3. It doesn't matter if the answer or the explanations in it are correct or not. If the answer has any gaps, give it a score of 0. Otherwise give it a score of 10.",
        "4. Make sure that the reason you gave for the score doesn't take correctness or accuracy into account, focusing only on gaps in the explanation."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

alternatives_metric_explicit = GEval(
    name="Lack of Alternatives Explicit",
    evaluation_steps=["1. Determine whether there probably aren't many other reasonable alternative explanations to the actual output for this input question.",
                      "2. Output a grade between 0 and 10 based on the above criterion.",
                      "3. A grade of 10 means there are probably no reasonable alternative explanations, and a grade of 0 means there are probably many reasonable alternatives explanations.",
                      "4. Do not take any other factors into account when giving a grade - correctness or accuracy should not affect the grade."
                      ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

articulation_metric_explicit = GEval(
    name="Articulation Explicit",
    evaluation_steps=[
        "1. Determine whether regardless of accuracy, this explanation is well articulated.",
        "2. Output a grade between 0 and 10 based on the above criterion.",
        "3. A grade of 10 means the explanation is perfectly articulated, while a grade of 0 means it is terribly articulated.",
        "4. Do not take any other factors into account when giving a grade - correctness or accuracy should not affect the grade."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

audience_awareness_metric_explicit = GEval(
    name="Audience Awareness Explicit",
    evaluation_steps=[
        "1. Determine whether the explanation is appropriate for a general audience with basic scientific knowledge (high school level).",
        "2. Score between 0 and 10: 10 means it explains well to a non-expert; 0 means it uses overly technical or inaccessible language.",
        "3. Ignore whether the content is factually correct—focus only on clarity and appropriateness of explanation level."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

perceived_truth_metric_explicit = GEval(
    name="Perceived Truth Explicit",
    criteria="You believe this explanation to be true.",
    evaluation_steps=[
      "1. You are an average person. Read the question and answer, and decide if you believe the answer or not",
      "2. Output a grade between 0 and 10 where 10 means you believe the explanation to be completely true, and 0 means that you think the explanation is completely false."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)



# Deprecated

# internal_coherence_metric = GEval(
#     name="Internal Coherence",
#     criteria="The parts of this explanation fit together coherently.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
# )

# completeness_metric = GEval(
#     name="Completeness",
#     criteria="There are no gaps in this explanation (actual output) for the given input question.",
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
# )


# alternatives_metric = GEval(
#     name="Lack of Alternatives",
#     criteria="There probably aren't many other reasonable alternative explanations to the actual output for this input question.",
#     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
# )

# articulation_metric = GEval(
#     name="Articulation",
#     criteria="Regardless of accuracy, this explanation is well articulated.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
# )

# audience_awareness_metric = GEval(
#     name="Audience Awareness",
#     criteria="The explanation is appropriate for a general audience with basic scientific knowledge (high school level).",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
# )

# perceived_truth_metric = GEval(
#     name="Perceived Truth",
#     criteria="You believe this explanation to be true.",
#     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
# )

