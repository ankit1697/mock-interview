import os, time, json
from typing import Dict, Any, List
from .loaders import relax_load_json, normalize_from_questions_block, pick_rubric_for_item
from .evaluator import evaluate_answer, call_openai
from .feedback import build_final_feedback, parse_feedback_sections

def run_full_pass(in_path: str, out_path: str) -> Dict[str, Any]:
    data = relax_load_json(in_path)
    items = normalize_from_questions_block(data)

    per_q: List[Dict[str, Any]] = []
    graded = 0
    t0 = time.time()
    mock_mode = os.getenv("MOCK_MODE","0") == "1"

    for idx, item in enumerate(items, 1):
        qid = item["id"]
        qtext = item["question"]
        ans = item["answer"]
        rubric = pick_rubric_for_item(item)

        res = evaluate_answer(qid, qtext, ans, rubric_json=rubric, ideal_answer=item.get("ideal_answer",""))

        fb = build_final_feedback(qtext, ans, res, call_openai)
        res["coach_feedback"] = {"raw": fb, "sections": parse_feedback_sections(fb)}
        res["question"] = qtext
        res["answer"] = ans
        res["type"] = item.get("type","")
        res["_original"] = item.get("_original", {})

        per_q.append(res)
        if isinstance(res.get("weighted_score"), (int, float)):
            graded += 1

        print(f"Processed {idx}/{len(items)}")

    nums = [x.get("weighted_score") for x in per_q if isinstance(x.get("weighted_score"), (int, float))]
    overall_numeric = round(sum(nums)/len(nums), 2) if nums else None

    out = {
        "meta": {
            "model": "gpt-4o",
            "mock_mode": mock_mode,
            "elapsed_sec": round(time.time()-t0, 2),
            "graded_questions": graded,
            "total_questions": len(items),
        },
        "per_question": per_q,
        "overall": {"numeric_mean_0_10": overall_numeric},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out
