"""
Simple local runner: executes root_agent against the test prompts and prints
the final synthesis output plus a basic execution trace (Section 25).

Usage (from the project root, with .env configured):
    python -m evaluation.evaluation                          # runs ALL scenarios
    python -m evaluation.evaluation --only small_business_smoke_test
"""
import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import InMemoryRunner
from google.genai import types

from agent import root_agent
from evaluation.test_cases import TEST_PROMPTS


async def run_scenario(prompt_name: str, prompt_text: str):
    print(f"\n{'=' * 80}\nSCENARIO: {prompt_name}\n{'=' * 80}")
    runner = InMemoryRunner(agent=root_agent, app_name="enterprise_strategist")
    session = await runner.session_service.create_session(
        app_name="enterprise_strategist", user_id="local_eval_user"
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt_text)])

    async for event in runner.run_async(
        user_id="local_eval_user", session_id=session.id, new_message=content
    ):
        author = getattr(event, "author", "?")
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts)
            print(f"\n--- [{author}] final response ---\n{text[:2000]}")
        elif getattr(event, "content", None) and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    print(f"[{author}] -> tool call: {part.function_call.name}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="Run just one scenario by name (see evaluation/test_cases.py), "
        "e.g. --only small_business_smoke_test",
    )
    args = parser.parse_args()

    cases = TEST_PROMPTS
    if args.only:
        cases = [c for c in TEST_PROMPTS if c["name"] == args.only]
        if not cases:
            names = ", ".join(c["name"] for c in TEST_PROMPTS)
            print(f"No scenario named '{args.only}'. Available: {names}")
            return

    for case in cases:
        await run_scenario(case["name"], case["prompt"])


if __name__ == "__main__":
    asyncio.run(main())
