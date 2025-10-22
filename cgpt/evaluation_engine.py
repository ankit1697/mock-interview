"""Interview evaluation utilities for step-by-step scoring.

Implements the Platform Evaluation Rubric and Business Logic (October 2025)
Dimensions: Technical Reasoning, Accuracy, Confidence, Problem-Solving, Flow/Stuckness.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

RUBRIC_WEIGHTS = {
    "technical_reasoning": 0.30,
    "accuracy": 0.25,
    "confidence": 0.15,
    "problem_solving": 0.15,
    "flow": 0.15,
}


def map_score_to_label(score: float) -> str:
    """Map a numeric score in [0, 5] to a qualitative label."""
    if score <= 1.0:
        return "Poor"
    if score <= 2.0:
        return "Fair"
    if score <= 3.0:
        return "Good"
    if score <= 4.0:
        return "Very Good"
    return "Excellent"


def aggregate_weighted_score(subscores: Dict[str, float]) -> float:
    """Compute the weighted aggregate score bounded to [0, 5]."""
    total = 0.0
    for key, weight in RUBRIC_WEIGHTS.items():
        value = float(subscores.get(key, 0.0))
        total += value * weight
    return max(0.0, min(5.0, total))


def default_heuristic_score(answer_text: str) -> Dict[str, float]:
    """Fallback heuristic scoring when no LLM is available."""
    text = (answer_text or "").strip()
    length = len(text.split())

    tech_keywords = [
        "model",
        "algorithm",
        "feature",
        "loss",
        "evaluate",
        "accuracy",
        "precision",
        "recall",
        "cluster",
        "k-means",
        "xgboost",
    ]
    tech_score = min(5.0, sum(1 for k in tech_keywords if k in text.lower()) * 1.2 + (0.01 * length))

    accuracy_score = 3.0 if length > 15 else 2.0 if length > 5 else 1.0

    hedges = ["maybe", "might", "could", "sort of", "i think", "probably", "not sure"]
    hedge_count = sum(1 for h in hedges if h in text.lower())
    confidence_score = max(1.0, 5.0 - hedge_count)

    step_words = ["first", "then", "next", "finally", "step", "approach", "process", "pipeline"]
    ps_score = min(5.0, sum(1 for w in step_words if w in text.lower()) * 1.3 + (0.005 * length))

    disfluencies = ["um", "uh", "ah", "erm"]
    disf_count = sum(1 for d in disfluencies if d in text.lower())
    flow_score = max(1.0, min(5.0, (length / 50.0) * 5.0 - disf_count))

    return {
        "technical_reasoning": round(tech_score, 2),
        "accuracy": round(accuracy_score, 2),
        "confidence": round(confidence_score, 2),
        "problem_solving": round(ps_score, 2),
        "flow": round(flow_score, 2),
    }


def _extract_json_candidate(text: str) -> Optional[Dict[str, float]]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None


def _score_with_llm(question: str, answer: str) -> Optional[Dict[str, float]]:
    if CLIENT is None:
        return None

    prompt_system = (
        "You are an expert interviewer evaluator. Score the candidate answer (0-5) on these "
        "dimensions: Technical Reasoning, Accuracy, Confidence, Problem-Solving, Flow (anti-stuckness). "
        "Return strict JSON with numeric scores and a short feedback string for each dimension."
    )
    prompt_user = f"Question: {question}\nAnswer: {answer}"

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:
        return None

    message = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_candidate(message)
    if not parsed:
        return None

    return {
        "technical_reasoning": float(
            parsed.get("Technical Reasoning", parsed.get("technical_reasoning", 0.0))
        ),
        "accuracy": float(parsed.get("Accuracy", parsed.get("accuracy", 0.0))),
        "confidence": float(parsed.get("Confidence", parsed.get("confidence", 0.0))),
        "problem_solving": float(
            parsed.get("Problem-Solving", parsed.get("problem_solving", 0.0))
        ),
        "flow": float(parsed.get("Flow", parsed.get("flow", 0.0))),
    }


def evaluate_interview_json(interview: Dict, use_llm: bool = True) -> Dict:
    """Evaluate an interview transcript represented as a JSON-like dict."""
    questions: List[Dict] = interview.get("questions", [])
    results: List[Dict] = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question", "")
        answer = item.get("answer", "")

        subscores = default_heuristic_score(answer)
        feedback_note = "Heuristic scoring used (no API key or use_llm=False)."

        if use_llm and CLIENT is not None:
            llm_scores = _score_with_llm(question, answer)
            if llm_scores:
                subscores = {
                    key: round(value, 2)
                    for key, value in llm_scores.items()
                }
                feedback_note = "LLM-assisted scoring applied."

        final_score = aggregate_weighted_score(subscores)
        rating = map_score_to_label(final_score)
        lowest_dimension = min(subscores.items(), key=lambda entry: entry[1])[0]
        recommendations = {
            "technical_reasoning": "Add more algorithmic detail, math, or code examples.",
            "accuracy": "Verify factual claims with sources or concrete numbers.",
            "confidence": "State conclusions decisively; reduce hedging language.",
            "problem_solving": "Outline a clear step-by-step approach or strategy.",
            "flow": "Reduce disfluencies and tighten the narrative structure.",
        }

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "subscores": subscores,
                "final_score": round(final_score, 3),
                "rating": rating,
                "feedback": feedback_note,
                "suggested_improvement": recommendations[lowest_dimension],
            }
        )

    overall_score = 0.0
    if results:
        overall_score = sum(item["final_score"] for item in results) / len(results)

    return {
        "interview_id": interview.get("interview_id"),
        "candidate_name": interview.get("candidate_name"),
        "results": results,
        "overall_score": round(overall_score, 3),
        "overall_rating": map_score_to_label(overall_score),
    }


__all__ = [
    "evaluate_interview_json",
    "aggregate_weighted_score",
    "map_score_to_label",
    "default_heuristic_score",
    "RUBRIC_WEIGHTS",
]
