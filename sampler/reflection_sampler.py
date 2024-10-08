import base64
import time
from typing import Any

import openai
from openai import OpenAI

from type_definitions import MessageList, SamplerBase

REFLECTION_SYSTEM_MESSAGE = '''You are a world-class AI system capable of complex reasoning and reflection. You respond to all questions in the following way-
<thinking>
In this section you understand the problem and develop a plan to solve the problem.

For easy problems-
Make a simple plan and use COT

For moderate to hard problems-
1. Devise a step-by-step plan to solve the problem. (don't actually start solving yet, just make a plan)
2. Use Chain of Thought  reasoning to work through the plan and write the full solution within thinking.

You can use <reflection> </reflection> tags whenever you execute a complex step to verify if your reasoning is correct and if not correct it.


</thinking>

<output>
In this section, provide the complete answer for the user based on your thinking process. Do not refer to the thinking tag. Include all relevant information and keep the response somewhat verbose, the user will not see what is in the thinking tag.
</output>'''


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "reflection_70b",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 6000,
    ):
        self.api_key_name = "test"
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"skip_special_tokens": False},
                    stream=False
                )
                try:
                    return response.choices[0].message.content.split("<output>")[1].split("</output>")[0]
                except:
                    return response.choices[0].message.content

            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
