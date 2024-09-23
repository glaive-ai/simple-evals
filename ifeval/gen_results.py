import os
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

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

GENERIC_SYSTEM_MESSAGE = "You are a helpful assistant. Provide clear and concise answers to the user's questions."

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

def generate_one(row, client, model_name, max_tokens, temperature, use_reflection):
    prompt = row["prompt"]
    system_message = REFLECTION_SYSTEM_MESSAGE if use_reflection else GENERIC_SYSTEM_MESSAGE
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_body={"skip_special_tokens": False},
        )
        
        output = response.choices[0].message.content
        return {
            "prompt": prompt,
            "response": output.split("<output>")[1].replace("</output>", "").strip() if use_reflection and "<output>" in output else output
        }
    except Exception as e:
        print(e)
        return None

def main(args):
    print("Starting evaluation process")
    print(f"Reflection mode: {'ON' if args.use_reflection else 'OFF'}")
    
    data = load_data(args.input_file)
    if not data:
        print("No data loaded. Exiting.")
        return

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.getenv("OPENAI_API_KEY", "test"),
    )

    responses = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_row = {executor.submit(generate_one, row, client, args.model_name, args.max_tokens, args.temperature, args.use_reflection): row for row in data}
        for future in tqdm(as_completed(future_to_row), total=len(data), desc="Generating responses"):
            result = future.result()
            if result is not None:
                responses.append(result)

    write_to_jsonl(responses, args.output_file)
    print("Evaluation process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI API script with customizable parameters")
    parser.add_argument("--input_file", default="data/ifeval_input_data.jsonl", help="Input file path")
    parser.add_argument("--output_file", default="data/reflection_output.jsonl", help="Output file path")
    parser.add_argument("--max_workers", type=int, default=128, help="Maximum number of worker threads")
    parser.add_argument("--model_name", default="glaiveai/Reflection-Llama-3.1-70B", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=6000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation")
    parser.add_argument("--base_url", default="http://0.0.0.0:5050/v1", help="Base URL for the API")
    parser.add_argument("--no-reflection", dest="use_reflection", action="store_false", help="Disable reflection mode (enabled by default)")
    parser.set_defaults(use_reflection=True)

    args = parser.parse_args()
    main(args)