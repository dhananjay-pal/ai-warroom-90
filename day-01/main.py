from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime
from pathlib import Path
import argparse

load_dotenv()
claude = Anthropic()
gpt = OpenAI()

def call_claude(prompt, max_tokens):
    start = time.time()
    msg = claude.messages.create(
        max_tokens = max_tokens,
        model = 'claude-sonnet-4-5',
        messages = [{
            "role":"user",
            "content":prompt
        }]
    )
    elapsed = time.time() - start
    return { 
        "modelname":msg.model, 
        "response":msg.content[0].text, 
        "input_tokens":msg.usage.input_tokens, 
        "output_tokens":msg.usage.output_tokens,
        "latency_seconds":elapsed
        }

def call_gpt(prompt, max_tokens):
    start = time.time()
    msg = gpt.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )
    elapsed = time.time() - start
    return { 
        "modelname":msg.model, 
        "response":msg.choices[0].message.content, 
        "input_tokens":msg.usage.prompt_tokens, 
        "output_tokens":msg.usage.completion_tokens,
        "latency_seconds":elapsed
        }


def log_run(prompt, results):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path =  log_dir/f"run_{timestamp}.json"
    data = {"prompt": prompt, "results": [results]}
    json_text = json.dumps(data, indent=2)
    log_path.write_text(json_text)
    print(f"\nLogged to {log_path}")


def loadVariables():
    load_dotenv()
    print('Anthropic key present:', bool(os.getenv('ANTHROPIC_API_KEY')))
    print('OpenAI key present:', bool(os.getenv('OPENAI_API_KEY')))

def main():
    parser = argparse.ArgumentParser(description="Dual-LLM CLI: Claude and GPT")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send")
    parser.add_argument("--model", type=str, choices=["claude","gpt","both"], default="both", help="Model to query")
    parser.add_argument("--max-tokens", type=int, default=500, help="Limit tokens")
    args = parser.parse_args()

    if args.model == "both":
        functions_to_call = [call_claude, call_gpt]
    elif args.model == "claude":
        functions_to_call = [call_claude]
    else:
        functions_to_call = [call_gpt]

    print(f"Prompt is: {args.prompt}")
    results = []
    for fn in functions_to_call:
        result = fn(args.prompt, args.max_tokens)
        results.append(result)
        print(f"[in: {result["input_tokens"]} tok | out: {result["output_tokens"]} tok | {result["latency_seconds"]} s]")
        
    log_run(args.prompt, results)
    # loadVariables()

if __name__ == "__main__":
    main()
# 