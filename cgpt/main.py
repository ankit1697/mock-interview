import os
import time
from typing import Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from evaluation_engine import evaluate_interview_json
from utils import read_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ICEBREAKER_COUNT = int(os.getenv("ICEBREAKER_COUNT", "2"))


def _format_dimension_name(dimension: str) -> str:
    parts = dimension.replace("_", " ").split()
    if parts and parts[0].lower() == "flow":
        return "Flow (anti-stuckness)"
    return " ".join(word.capitalize() for word in parts)


def display_incremental_evaluation(evaluation: Dict, latest_question_id: int) -> None:
    """Print the most recent question score and running aggregate."""
    results: List[Dict] = evaluation.get("results", [])
    if not results:
        return

    latest = next((entry for entry in results if entry.get("id") == latest_question_id), None)

    if latest is None:
        print("\nEvaluation Update:")
        print(f"Question {latest_question_id} has not been evaluated yet.")
        return

    print("\nEvaluation Update:")
    if latest.get("skipped"):
        print(f"Question {latest_question_id} was an icebreaker and was not scored.")
    else:
        print(f"Question {latest_question_id} rating: {latest['final_score']} ({latest['rating']})")
        for key, value in latest["subscores"].items():
            label = _format_dimension_name(key)
            print(f"  {label}: {value}")
        print(f"  Feedback note: {latest['feedback']}")
        print(f"  Technical feedback: {latest.get('technical_feedback')}")
        print(f"  Suggested improvement: {latest['suggested_improvement']}")
        example_answer = latest.get("example_answer")
        if example_answer:
            print("  Example answer:")
            print(f"    {example_answer}")

    overall_score = evaluation.get("overall_score")
    overall_rating = evaluation.get("overall_rating")
    if overall_score is None:
        print("Overall so far: Not enough evaluated questions yet.")
    else:
        print(f"Overall so far: {overall_score} ({overall_rating})")


def display_final_summary(evaluation: Dict) -> None:
    """Print a concise summary after the interview ends."""
    results: List[Dict] = evaluation.get("results", [])
    if not results:
        print("No answers recorded; no evaluation generated.")
        return

    print("\nFinal Evaluation Summary:")
    overall_score = evaluation.get("overall_score")
    overall_rating = evaluation.get("overall_rating")
    if overall_score is not None:
        print(f"Overall score: {overall_score} ({overall_rating})")
    else:
        print("Overall score: Not rated (only icebreakers answered).")

    for entry in results:
        if entry.get("skipped"):
            continue
        print(
            f"- Question {entry['id']}: {entry['final_score']} ({entry['rating']})"
        )

    overall_feedback = evaluation.get("overall_feedback") or {}
    summary_text = overall_feedback.get("summary")
    if summary_text:
        print(f"\nOverall feedback: {summary_text}")

    improvement_items = overall_feedback.get("areas_for_improvement") or []
    if improvement_items:
        print("Areas for improvement:")
        for item in improvement_items:
            print(f"- {item}")


def generate_question(api_client: OpenAI, messages: List[Dict[str, str]]) -> str:
    start = time.time()
    response = api_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    elapsed = round(time.time() - start, 2)
    question = response.choices[0].message.content.strip()
    print(f"(Question generation took {elapsed} seconds)")
    return question


def _append_question_prompt(
    messages: List[Dict[str, str]],
    icebreakers_remaining: int,
    asked_icebreakers: int,
) -> None:
    if icebreakers_remaining > 0:
        index = asked_icebreakers + 1
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Ask icebreaker question #{index} to build rapport. Keep it friendly, "
                    "brief, and avoid technical deep dives."
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Ask the next role-specific technical interview question. "
                    "Focus on evaluating senior data science skills one topic at a time."
                ),
            }
        )


def _question_label(question_number: int) -> str:
    if question_number <= ICEBREAKER_COUNT:
        return f"Icebreaker {question_number}"
    return f"Question {question_number}"


def request_next_question(
    api_client: OpenAI,
    messages: List[Dict[str, str]],
    icebreakers_remaining: int,
    asked_icebreakers: int,
) -> Tuple[str, int, int]:
    _append_question_prompt(messages, icebreakers_remaining, asked_icebreakers)
    question = generate_question(api_client, messages)
    messages.append({"role": "assistant", "content": question})

    if icebreakers_remaining > 0:
        icebreakers_remaining -= 1
        asked_icebreakers += 1

    return question, icebreakers_remaining, asked_icebreakers


def main() -> None:
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    api_key = api_key.strip() if api_key else None

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the interviewer.")

    client = OpenAI(api_key=api_key)

    resume_path = os.path.join(BASE_DIR, "resume.pdf")
    jd_path = os.path.join(BASE_DIR, "jd.txt")
    resume_text = read_file(resume_path)
    job_description = read_file(jd_path)

    system_prompt = (
        "You are an expert AI interviewer for senior data science roles. "
        "Begin the interview with friendly icebreaker questions (two maximum) before switching to "
        "role-specific technical topics. Use the provided resume and job description to tailor "
        "each technical question. Ask exactly one question at a time and respond only with the "
        "next question after each answer. Do not provide commentary or evaluation.\n\n"
        f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    interview_id = f"session_{int(time.time())}"
    candidate_name = os.getenv("CANDIDATE_NAME", "Candidate")
    transcript: List[Dict] = []

    icebreakers_remaining = ICEBREAKER_COUNT
    asked_icebreakers = 0

    current_question, icebreakers_remaining, asked_icebreakers = request_next_question(
        client,
        messages,
        icebreakers_remaining,
        asked_icebreakers,
    )
    if not current_question:
        raise RuntimeError("Failed to generate the opening question.")

    question_counter = 1
    print(f"\n{_question_label(question_counter)}: {current_question}\n")

    latest_evaluation: Dict = {}

    while True:
        answer = input("Your Answer (or type 'exit'): ").strip()
        if answer.lower() == "exit":
            break

        transcript.append(
            {
                "id": question_counter,
                "question": current_question,
                "answer": answer,
                "is_icebreaker": question_counter <= ICEBREAKER_COUNT,
            }
        )

        messages.append({"role": "user", "content": answer})

        evaluation_payload = {
            "interview_id": interview_id,
            "candidate_name": candidate_name,
            "questions": transcript,
            "icebreaker_count": ICEBREAKER_COUNT,
        }

        latest_evaluation = evaluate_interview_json(evaluation_payload, use_llm=True)
        display_incremental_evaluation(latest_evaluation, question_counter)

        question_counter += 1
        next_question, icebreakers_remaining, asked_icebreakers = request_next_question(
            client,
            messages,
            icebreakers_remaining,
            asked_icebreakers,
        )
        if not next_question:
            print("No further questions generated. Ending interview.")
            break

        current_question = next_question
        print(f"\n{_question_label(question_counter)}: {current_question}\n")

    if transcript and latest_evaluation:
        display_final_summary(latest_evaluation)
    else:
        print("Interview ended without any recorded answers.")


if __name__ == "__main__":
    main()