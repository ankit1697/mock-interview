"""Interview evaluation utilities for step-by-step scoring.

Implements the Platform Evaluation Rubric and Business Logic (October 2025)
Dimensions: Technical Reasoning, Accuracy, Confidence, Problem-Solving, Flow/Stuckness.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

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

SKIP_NOTE = "Icebreaker question — skipped for scoring."

REFERENCE_EVALUATION_EXAMPLE = """{
    "interview_id": "mock_001",
    "candidate_name": "Anuja Tipare",
    "job_role": "Data Scientist",
    "interview_date": "2025-05-21",
    "questions": [
        {
            "id": 1,
            "question": "Hi Anuja, how are you?",
            "answer": "I'm doing good. How are you?",
            "feedback": "This is just an introduction question. No feedback needed.",
            "score": null,
            "improvement_areas": [],
            "type": "introduction",
            "followup_to": null
        },
        {
            "id": 2,
            "question": "What we'll do is we'll have a few questions from the resume from any of the projects that you worked on. We'll try to focus on the basics of machine learning data science. So to start off with, describe any one particular project that you've worked on recently, primarily involving some form of data science or ML if possible, any particular project and you can go from the problem statement to what you did and how, you know, sort of what the results were or if you had, if you created some business impact.",
            "answer": "Definitely Ankit. So last quarter I worked on a recommendation system for a bookstore. So the problem statement was that the bookstore currently was recommending books by top selling items and it wasn't personalized for the user. So what I and my team did was we created a personalized recommendation system which was based on the recency, frequency and monetary value of the data that we got from. So the data had around 30 book categories for 33k customers and we had the recency, frequency, monetary values and as it is very popular in the marketing segment, we used the RFM to create clusters. So we did two kinds of clustering. The first was the customer value segmentation based on the RFM values which basically gave me four clusters of my customers. Some were the potential loyalists, some were the champions who had very good high RFM values and there were lapsed customers who had low RFM values and that was the first clustering that we did. Within the group style, just explain me, did clustering based on the book categories. Now, as I said, we had somewhere around 30 book categories, but we clubbed them together like history and something subjective was clubbed together. So based on this, we did another clustering, both of which was done using k-means. Once the clusters were ready, we implemented collaborative filtering for both user-based and item-based filtering. To get to see how the personalized recommendation system worked, we implemented an A-B testing framework. So we had a test group and a control group and then we tracked some KPIs like the click through rate, conversion rate and the average session time. So the metrics over the silhouette score, which was 0.44 were kind of okay. So it performed moderately. Yeah, that was the project that I did.",
            "feedback": "You communicated your project experience with confidence and covered the full lifecycle from problem statement to evaluation. This is a strong example of how to describe a project.",
            "score": 8.5,
            "improvement_areas": ["Discuss trade-offs or challenges faced","Explain terms like RFM, collaborative filtering, and silhouette score briefly for clarity"],
            "type": "technical",
            "followup_to": null
        },
        {
            "id": 3,
            "question": "Great. So it sounds like that you had like a two-step approach. First, you were clustering the users and then you were clustering the books and the items, like the purchase. Okay. So you used k-means that I can see. Can you just tell me, so when we do k-means clustering, we prioritize normalizing the features or scaling the features. Do you know why is that important?",
            "answer": "So how k-means works is that randomly centroids are allocated between the data points and then it just calculates the mean of the group, the groups that are present in the centroids and then it gradually updates. Now if there are certain outliers, it is almost very much possible that your centroids or your means will get, they will be disrupted. It will be more inclined towards the outline and we don't want that. So for that purpose, it is necessary that we always normalize or bring it to a uniform scale before we perform k-means.",
            "feedback": "You rightly mentioned that unequal feature scales and outliers can skew centroids, which is a key insight. However, the explanation could benefit from clearer phrasing and a more direct focus on why scaling is important, specifically, because k-means relies on Euclidean distance, which is sensitive to feature magnitudes. You can also give an example here, like having features such as height in meters and weight in grams. Without scaling, the weight in grams would completely overshadow the height in meters in any distance calculation.",
            "score": 7,
            "improvement_areas": ["Emphasize that k-means being sensitive to feature scale",""],
            "type": "technical",
            "followup_to": 2
        }
    ],
    "overall_feedback": {
        "overall_score": 6.8,
        "scores": {
            "technical knowledge": 8,
            "communication"      : 8,
            "problem solving"    : 7.5,
            "behavioral"         : 8.5
        },
        "strengths": [
            "Demonstrates strong foundational and articulation understanding of ML, clustering, evaluation metrics, and hypothesis testing",
            "Shows genuine interest in learning, especially in new areas like LLMs",
            "Practical Application Focus",
            "Strong Project Storytelling",
            "Behavioral responses reflect team orientation and stakeholder awareness"
        ],
        "areas_of_improvement": [
            "Improve technical precision and structure in explanations (e.g., bias-variance, precision-recall)",
            "Use formal definitions and formulas where relevant to add depth",
            "Emphasizing Personal Contribution and growth in Behavioral Answers",
            "Avoid filler phrases; aim for concise, confident delivery"
        ]
    }
}"""


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


def _fallback_technical_feedback(subscores: Dict[str, float]) -> str:
    weakest = min(subscores.items(), key=lambda entry: entry[1])[0]
    mapping = {
        "technical_reasoning": (
            "Dive deeper into the core mechanics and trade-offs. Reference concrete models,"
            " data treatments, or evaluation math to strengthen the narrative."
        ),
        "accuracy": (
            "Ground the answer with verifiable metrics, benchmarks, or validation steps to"
            " demonstrate rigor."
        ),
        "confidence": (
            "Deliver conclusions decisively and reduce hedging; highlight prior successes to"
            " anchor your stance."
        ),
        "problem_solving": (
            "Lay out the end-to-end plan explicitly—define the problem, outline alternatives,"
            " and justify the chosen path."
        ),
        "flow": (
            "Tighten the storytelling arc. Remove filler words and use transitions to keep the"
            " interviewer oriented."
        ),
    }
    return mapping.get(weakest, "Provide more detail where the interviewer is probing.")


def _fallback_example_answer(question: str) -> str:
    prompt = question.strip()
    if not prompt:
        return (
            "A strong response should frame the challenge, detail the technical approach,"
            " highlight tooling, and finish with measurable impact."
        )
    return (
        "A high-scoring answer would: 1) frame the business or research context for the question"
        "; 2) outline the analytical or modeling approach with tooling choices; 3) discuss key"
        " metrics and validation; and 4) finish with measurable results or lessons learned."
    )


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


def _score_and_feedback_with_llm(question: str, answer: str) -> Optional[Dict[str, Any]]:
    if CLIENT is None:
        return None

    prompt_system = (
        "You are an expert interviewer evaluator for senior data science roles. For the provided"
        " question and answer you must return strict JSON with this schema: {\n"
        "  \"scores\": {\n"
        "    \"technical_reasoning\": float 0-5,\n"
        "    \"accuracy\": float 0-5,\n"
        "    \"confidence\": float 0-5,\n"
        "    \"problem_solving\": float 0-5,\n"
        "    \"flow\": float 0-5\n"
        "  },\n"
        "  \"technical_feedback\": string (two to three sentences highlighting strengths and actionable improvements),\n"
        "  \"example_answer\": string (a high-quality sample answer with clear structure and technical detail)\n"
        "}.\n"
        "Ensure the JSON is valid, without markdown fences, and keep the example answer grounded in the question.\n\n"
        "Reference evaluation for tone and structure guidance:\n"
    ) + REFERENCE_EVALUATION_EXAMPLE
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

    scores = parsed.get("scores", parsed)
    result: Dict[str, Any] = {
        "scores": {
            "technical_reasoning": float(
                scores.get("technical_reasoning", scores.get("Technical Reasoning", 0.0))
            ),
            "accuracy": float(scores.get("accuracy", scores.get("Accuracy", 0.0))),
            "confidence": float(scores.get("confidence", scores.get("Confidence", 0.0))),
            "problem_solving": float(
                scores.get("problem_solving", scores.get("Problem-Solving", 0.0))
            ),
            "flow": float(scores.get("flow", scores.get("Flow", 0.0))),
        },
        "technical_feedback": parsed.get("technical_feedback")
        or parsed.get("Technical Feedback"),
        "example_answer": parsed.get("example_answer")
        or parsed.get("Example Answer"),
    }

    return result


def _summarize_session_with_llm(
    results: List[Dict[str, Any]],
    candidate_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if CLIENT is None or not results:
        return None

    payload = {
        "candidate_name": candidate_name or "the candidate",
        "responses": [
            {
                "question": item.get("question"),
                "answer": item.get("answer"),
                "final_score": item.get("final_score"),
                "rating": item.get("rating"),
                "technical_feedback": item.get("technical_feedback"),
            }
            for item in results
        ],
    }

    prompt_system = (
        "You are an AI interview coach. Given scored interview responses, craft a concise"
        " final evaluation: summarize overall performance with a constructive, strengths-first"
        " tone and enumerate concrete focus areas for improvement. Respond with strict JSON"
        " containing keys 'summary' (2-3 sentences) and 'areas_for_improvement' (array of 2-4"
        " short bullet-worthy strings).\n\n"
        "Reference evaluation for target voice and specificity:\n"
    ) + REFERENCE_EVALUATION_EXAMPLE
    prompt_user = json.dumps(payload)

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    except Exception:
        return None

    message = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_candidate(message)
    if not parsed:
        return None

    areas = parsed.get("areas_for_improvement") or parsed.get("AreasForImprovement")
    if isinstance(areas, str):
        areas = [areas]

    return {
        "summary": parsed.get("summary") or parsed.get("Summary"),
        "areas_for_improvement": areas or [],
    }


def _compile_overall_feedback(
    results: List[Dict[str, Any]],
    candidate_name: Optional[str],
    use_llm: bool,
) -> Dict[str, Any]:
    if use_llm:
        llm_summary = _summarize_session_with_llm(results, candidate_name)
        if llm_summary:
            return llm_summary

    if not results:
        return {
            "summary": "No evaluative questions recorded; unable to compute overall feedback.",
            "areas_for_improvement": [],
        }

    avg_score = sum(item.get("final_score", 0.0) for item in results) / len(results)
    strongest = max(results, key=lambda item: item.get("final_score", 0.0))
    weakest = min(results, key=lambda item: item.get("final_score", 0.0))

    summary = (
        f"Overall performance is rated {map_score_to_label(avg_score)} with an average score of"
        f" {avg_score:.2f}. Maintain the strengths shown on '{strongest.get('question', 'key topics')}'."
    )
    areas = [
        f"Revisit '{weakest.get('question', 'weaker topics')}' incorporating: {weakest.get('technical_feedback')}"
    ]

    return {
        "summary": summary,
        "areas_for_improvement": areas,
    }


def evaluate_interview_json(interview: Dict, use_llm: bool = True) -> Dict:
    """Evaluate an interview transcript represented as a JSON-like dict."""
    questions: List[Dict] = interview.get("questions", [])
    icebreaker_count = int(interview.get("icebreaker_count", 0))
    results: List[Dict] = []

    for index, item in enumerate(questions, start=1):
        qid = item.get("id", index)
        question = item.get("question", "")
        answer = item.get("answer", "")
        is_icebreaker = bool(item.get("is_icebreaker")) or (index <= icebreaker_count)

        if is_icebreaker:
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": answer,
                    "skipped": True,
                    "skip_reason": SKIP_NOTE,
                    "technical_feedback": SKIP_NOTE,
                    "example_answer": None,
                    "subscores": {},
                    "final_score": None,
                    "rating": None,
                    "feedback": SKIP_NOTE,
                    "suggested_improvement": None,
                }
            )
            continue

        subscores = default_heuristic_score(answer)
        technical_feedback = _fallback_technical_feedback(subscores)
        example_answer = _fallback_example_answer(question)
        feedback_note = "Heuristic scoring used (no API key or use_llm=False)."

        if use_llm and CLIENT is not None:
            llm_result = _score_and_feedback_with_llm(question, answer)
            if llm_result:
                llm_scores = llm_result.get("scores", {})
                subscores = {
                    key: round(float(value), 2)
                    for key, value in llm_scores.items()
                }
                technical_feedback = llm_result.get("technical_feedback", technical_feedback)
                example_answer = llm_result.get("example_answer", example_answer)
                if isinstance(technical_feedback, list):
                    technical_feedback = " ".join(str(part) for part in technical_feedback)
                if isinstance(example_answer, list):
                    example_answer = " ".join(str(part) for part in example_answer)
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
                "technical_feedback": technical_feedback,
                "example_answer": example_answer,
                "skipped": False,
            }
        )

    evaluated_results = [item for item in results if not item.get("skipped")]
    overall_score = 0.0
    overall_rating = "Not Rated"
    if evaluated_results:
        overall_score = sum(item["final_score"] for item in evaluated_results) / len(evaluated_results)
        overall_rating = map_score_to_label(overall_score)

    overall_feedback = _compile_overall_feedback(
        evaluated_results,
        candidate_name=interview.get("candidate_name"),
        use_llm=use_llm,
    ) if evaluated_results else None

    return {
        "interview_id": interview.get("interview_id"),
        "candidate_name": interview.get("candidate_name"),
        "results": results,
        "overall_score": round(overall_score, 3) if evaluated_results else None,
        "overall_rating": overall_rating,
        "overall_feedback": overall_feedback,
    }


def format_answer_evaluation_feedback(evaluation_result: Dict[str, Any]) -> str:
    """Format a single answer evaluation result into human-readable feedback text."""
    if not evaluation_result or not evaluation_result.get("results"):
        return "Answer recorded. Evaluation processing..."
    
    result = evaluation_result["results"][0]
    feedback_parts = []
    
    feedback_parts.append(f"Rating: {result.get('rating', 'N/A')} (Score: {result.get('final_score', 0)}/5)")
    
    subscores = result.get('subscores', {})
    if subscores:
        feedback_parts.append("\nDimension Scores:")
        for dim, score in subscores.items():
            dim_name = dim.replace('_', ' ').title()
            feedback_parts.append(f"  - {dim_name}: {score}/5")
    
    tech_feedback = result.get('technical_feedback')
    if tech_feedback:
        feedback_parts.append(f"\nFeedback: {tech_feedback}")
    
    improvement = result.get('suggested_improvement')
    if improvement:
        feedback_parts.append(f"\nSuggested Improvement: {improvement}")
    
    return "\n".join(feedback_parts)


def format_interview_summary(evaluation_result: Dict[str, Any]) -> str:
    """Format the complete interview evaluation result into a human-readable summary."""
    if not evaluation_result:
        return "No evaluation data available."
    
    summary_parts = []
    
    overall_score = evaluation_result.get("overall_score")
    overall_rating = evaluation_result.get("overall_rating")
    
    if overall_score:
        summary_parts.append(f"\nOverall Score: {overall_score}/5.0 ({overall_rating})")
    
    overall_feedback = evaluation_result.get("overall_feedback", {})
    if overall_feedback:
        summary_text = overall_feedback.get("summary")
        if summary_text:
            summary_parts.append(f"\n{summary_text}")
        
        areas = overall_feedback.get("areas_for_improvement", [])
        if areas:
            summary_parts.append("\nAreas for Improvement:")
            for area in areas:
                summary_parts.append(f"  • {area}")
    
    return "\n".join(summary_parts) if summary_parts else "No summary available."


__all__ = [
    "evaluate_interview_json",
    "aggregate_weighted_score",
    "map_score_to_label",
    "default_heuristic_score",
    "format_answer_evaluation_feedback",
    "format_interview_summary",
    "RUBRIC_WEIGHTS",
]
