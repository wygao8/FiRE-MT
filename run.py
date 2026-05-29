import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from scripts.fire_eval import PREFERENCES, run_eval, save_and_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-language", required=True)
    parser.add_argument("--tgt-language", required=True)
    parser.add_argument("--dataset", default="all", choices=["ranked", "tied", "all"])
    parser.add_argument("--mode", required=True, choices=["api", "vllm"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--preferences", nargs="+", default=PREFERENCES)
    parser.add_argument("--temperature", type=float, default=0.6)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "api" and not (args.api_key and args.api_url):
        print("ERROR: --api-key and --api-url are required for --model api")
        sys.exit(1)
    if args.mode == "vllm" and not args.model_path:
        print("ERROR: --model-path is required for --model vllm")
        sys.exit(1)

    outputs = run_eval(
        src_language=args.src_language,
        tgt_language=args.tgt_language,
        dataset_type=args.dataset,
        mode=args.mode,
        model_name=args.model_name,
        preferences=args.preferences,
        temperature=args.temperature,
        api_key=args.api_key,
        api_url=args.api_url,
        model_path=args.model_path,
    )

    save_and_score(outputs, args.src_language, args.tgt_language, args.model_name)