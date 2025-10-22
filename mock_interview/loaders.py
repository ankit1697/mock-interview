import json, re
from typing import List, Dict, Any
from .rubrics import TECH_RUBRIC, BEHAVIORAL_RUBRIC

def relax_load_json(path: str) -> Any:
    txt = open(path, "r", encoding="utf-8").read()
    txt = re.sub(r",\s*([\]\}])", r"\1", txt)
    return json.loads(txt)

def normalize_from_questions_block(data: dict) -> List[dict]:
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Top-level 'questions' list not found.")
    out = []
    for i, d in enumerate(data["questions"], 1):
        out.append({
            "id": d.get("id", f"q{i}"),
            "question": d.get("question", ""),
            "answer": d.get("answer", ""),
            "type": d.get("type", ""),
            "ideal_answer": d.get("ideal_answer", ""),
            "resources": d.get("resources", []),
            "concept_weights_json": d.get("concept_weights_json", {}),
            "_original": {
                "feedback": d.get("feedback"),
                "score": d.get("score"),
                "improvement_areas": d.get("improvement_areas"),
                "followup_to": d.get("followup_to")
            }
        })
    return out

def pick_rubric_for_item(item: dict) -> Dict[str, Any]:
    t = (item.get("type") or "").lower()
    if "behav" in t or "soft" in t:
        return BEHAVIORAL_RUBRIC
    return TECH_RUBRIC
