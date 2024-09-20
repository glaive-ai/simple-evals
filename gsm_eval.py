import random
import re

import blobfile as bf
import pandas

from datasets import load_dataset


import common
from common import ANSWER_PATTERN, HTML_JINJA, check_equality
from type_definitions import Eval, EvalResult, SamplerBase, SingleEvalResult


QUERY_TEMPLATE = """{question}. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem."""


class GSMEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: int | None = None):
        examples = load_dataset("openai/gsm8k","main",split="test")
        self.examples = examples
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            matches = re.findall(ANSWER_PATTERN, response_text)
            if not matches:
                extracted_answer = ""
                score = 0.0
            else:
                extracted_answer = matches[-1]
                correct_answer = row["answer"].split('####')[1].strip().replace("%","")
                score = float(check_equality(self.equality_checker, correct_answer, extracted_answer.replace("%","")))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
