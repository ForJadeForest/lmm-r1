import json
import os
import random
import re
from argparse import ArgumentParser

from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from loguru import logger
from concurrent import futures

app = Flask(__name__)

problem_to_answer = {}
enable_format_reward = True


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>", "<end_of_turn>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1


def verify_math(content, sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            reward = 0.0
            print("Failed to verify: ", e)
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)
    return reward


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    if "query" not in data or "prompts" not in data or "labels" not in data:
        return jsonify({"error": "query, prompts, and labels fields are required"}), 400
    rewards = []
    format_rewards = []
    acc_rewards_futures = []
    for q, problem, answer in zip(data["query"], data["prompts"], data["labels"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if not answer.startswith("$"):
            answer = "$" + answer + "$"

        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        # Apply format reward only if enabled
        format_reward = 0.0
        if enable_format_reward:
            format_reward = float(verify_format(response)) * 0.5
        acc_reward_future = math_verify_executor.submit(verify_math, response, answer)

        do_print = random.randint(1, 20) == 1
        if do_print:
            info = (
                "\n" + "="*80 + "\n"
                f"[Query]:\n{q}\n\n"
                f"[Problem]:\n{problem}\n\n"
                f"[Ground Truth Answer]:\n{answer}\n\n"
                f"[Model Response]:\n{response}\n\n"
                f"[Rewards]:\n"
                f"  • Format Score: {format_reward:.2f}\n"
                f"  • Accuracy Score: {acc_reward_future.result():.2f}\n"
                "="*80 + "\n"
            )
            info = re.sub(r"<\|.*?\|>|<pad>", "", info)
            logger.info(info)

        format_rewards.append(format_reward)
        acc_rewards_futures.append(acc_reward_future)
    acc_rewards = [f.result() for f in acc_rewards_futures]
    rewards = [f + a for f, a in zip(format_rewards, acc_rewards)]
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards, "format_rewards": format_rewards, "acc_rewards": acc_rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt-template", type=str, default=None, help="Prompt template", required=True)
    parser.add_argument("--input_key", type=str, default="prompt", help="The key name of prompt.")
    parser.add_argument("--log_file", type=str, default="remote_rm.log", help="Log file path")
    parser.add_argument(
        "--disable-format-reward",
        action="store_true",
        help="Disable format reward calculation. When enabled (default), responses get +0.5 reward for correct format.",
    )
    args = parser.parse_args()
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    logger.remove()
    logger.add(args.log_file)

    # Set format reward flag based on command line argument
    enable_format_reward = not args.disable_format_reward
    print(f"Format reward is {'disabled' if args.disable_format_reward else 'enabled'}")
    logger.info(f"Format reward is {'disabled' if args.disable_format_reward else 'enabled'}")

    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template == "chatml":
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        response_prefix = r"Assistant: "
    elif args.prompt_template == "phi3":
        response_prefix = r"<|assistant|>\n"
    elif args.prompt_template == "phi4":
        response_prefix = r"<|assistant|>\n"
    elif args.prompt_template == "gemma3":
        response_prefix = r"<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")

    # math_verify can only run in main thread
    math_verify_executor = futures.ProcessPoolExecutor(max_workers=16)

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    math_verify_executor.shutdown()
