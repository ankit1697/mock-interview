#!/usr/bin/env python
import os, argparse, json
from dotenv import load_dotenv
from mock_interview.runner import run_full_pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to session JSON with a top-level 'questions' list")
    ap.add_argument("--out", help="Output path (defaults to <input>_scored.json)")
    ap.add_argument("--mock", action="store_true", help="Enable MOCK_MODE=1 (no API calls)")
    args = ap.parse_args()

    load_dotenv()
    if args.mock:
        os.environ["MOCK_MODE"] = "1"

    out = args.out or (os.path.splitext(args.input)[0] + "_scored.json")
    result = run_full_pass(args.input, out)
    print(json.dumps({"saved": out, "graded": result["meta"]["graded_questions"]}, indent=2))

if __name__ == "__main__":
    main()
