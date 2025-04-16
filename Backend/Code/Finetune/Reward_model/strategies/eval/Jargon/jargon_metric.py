
import pandas as pd
import logging
from .jargon_util import calculate_grade
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class JargonMetric(BaseMetric):
    def __init__(
            self,
            threshold: float = 0.5,
            # Optional
    ):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = calculate_grade(test_case.actual_output)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Jargon Metric"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main_logger")



def answer_generator_from_csv(csv_path, indexes):
    df = pd.read_csv(csv_path)
    df = df.where(df["index"].isin(indexes)).dropna()
    return df


def add_answer_grades(answers_df: pd.DataFrame):
    answers_df["dejargonizer_report"] = None
    results = [calculate_grade(row["answer"]) for index, row in answers_df.iterrows()]
    answers_df["dejargonizer_report"] = results
    return answers_df


def main():
    random_sample = [3898, 8916, 2136, 6061, 7766]
    answers = answer_generator_from_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/DPO_data/llama3_18B_ask_science_answers.csv", random_sample)
    answers = add_answer_grades(answers)

    answers.to_csv("/Users/mattan.yeroushalmi/studies/thesis/Benchmarking/deep_eval/Jargon/llama3_18B_ask_science_jargon_index.csv")



if __name__ == "__main__":
    main()