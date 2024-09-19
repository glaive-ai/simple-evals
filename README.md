# Overview

This repository is a fork of https://github.com/openai/simple-evals, used for running benchmarks on the Reflection 70B model.

## Running locally

### For HumanEval

```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

### For GPQA

* Download this csv file in the directory: https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_main.csv 
  Note: You'll need to be authenticated with huggingface hub and accepted the conditions on the repo.
  ```bash
  wget --header="Authorization: Bearer <hf_token>" https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_main.csv
  ```

### Run benchmarking

1. Install requirements: `pip3 install -r requirements`
2. Start vllm server locally: `vllm serve glaiveai/Reflection-Llama-3.1-70B --host 0.0.0.0 --port 5050 --tensor-parallel 8`
3. Set OPENAI_API_KEY env var for running the equality checker.
4. Run `python3 run_reflection_eval.py`

### Running ifeval

```bash
cd ifeval
python3 gen_results.py
python3 -m evaluation_main \
  --input_data=./data/ifeval_input_data.jsonl \
  --input_response_data=./data/reflection_output.jsonl \
  --output_dir=./data/
```