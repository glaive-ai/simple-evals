import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        return []

def write_to_jsonl(data, filename):
    try:
        with open(filename, 'w') as file:
            for item in data:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
    except Exception as e:
        print(e)



# Configuration
INPUT_FILE = "data/ifeval_input_data.jsonl"
OUTPUT_FILE = "data/reflection_output.jsonl"
MAX_WORKERS = 128
MODEL_NAME = ""
MAX_TOKENS = 6000
TEMPERATURE = 0.0

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://0.0.0.0:5050/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "test"),
)

def generate_one(row):
    prompt = row["prompt"]
    system = '''You are a world-class AI system capable of complex reasoning and reflection. You respond to all questions in the following way-
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
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            extra_body={"skip_special_tokens": False},
        )
        
        output = response.choices[0].message.content
        return {
            "prompt": prompt,
            "response": output.split("<output>")[1].replace("</output>", "").strip() if "<output>" in output else output
        }
    except Exception as e:
        return None

def main():
    print("Starting evaluation process")
    
    data = load_data(INPUT_FILE)
    if not data:
        print("No data loaded. Exiting.")
        return

    responses = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(generate_one, row): row for row in data}
        for future in tqdm(as_completed(future_to_row), total=len(data), desc="Generating responses"):
            result = future.result()
            if result is not None:
                responses.append(result)

    write_to_jsonl(responses, OUTPUT_FILE)
    print("Evaluation process completed")

if __name__ == "__main__":
    main()