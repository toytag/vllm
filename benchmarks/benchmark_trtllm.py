"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset if len(data["conversations"]) >= 2]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

def run_trtllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    max_batch_size: int,
) -> float:
    # llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    # tokenizer.pad_token = tokenizer.eos_token
    from optimum.nvidia.pipelines import pipeline
    llm = pipeline("text-generation", model=model)
    prompts = [prompt for prompt, _, _ in requests]

    # pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    # batch: List[str] = []
    # max_prompt_len = 0
    # max_output_len = 0
    # for i in range(len(requests)):
    #     prompt, prompt_len, output_len = requests[i]
    #     # Add the prompt to the batch.
    #     batch.append(prompt)
    #     max_prompt_len = max(max_prompt_len, prompt_len)
    #     max_output_len = max(max_output_len, output_len)
    #     if len(batch) < max_batch_size and i != len(requests) - 1:
    #         # Check if we can add more requests to the batch.
    #         _, next_prompt_len, next_output_len = requests[i + 1]
    #         if (max(max_prompt_len, next_prompt_len) +
    #                 max(max_output_len, next_output_len)) <= 2048:
    #             # We can add more requests to the batch.
    #             continue

    #     # Generate the sequences.
    #     input_ids = tokenizer(batch, return_tensors="pt",
    #                           padding=True).input_ids
    #     llm_outputs = llm.generate(
    #         input_ids=input_ids.cuda(),
    #         max_new_tokens=max_output_len,
    #     )
    #     # Include the decoding time.
    #     tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
    #     pbar.update(len(batch))

    #     # Clear the batch.
    #     batch = []
    #     max_prompt_len = 0
    #     max_output_len = 0
    llm(prompts, max_length=128, padding=True)

    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(0)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.backend == "trtllm":
        elapsed_time = run_trtllm(requests, args.model, tokenizer, args.batch_size)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["trtllm"],
                        default="trtllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=128,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1024,
                        help="Number of prompts to process.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=64)
    args = parser.parse_args()
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
    main(args)
