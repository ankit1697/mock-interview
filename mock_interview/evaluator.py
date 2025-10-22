import os, json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = None

def _client_openai():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key and os.getenv("MOCK_MODE","0") != "1":
            raise RuntimeError("OPENAI_API_KEY not set; use .env or set MOCK_MODE=1")
        _client = OpenAI(api_key=key) if key else None
    return _client

NUMERIC_SCHEMA = {
  "rubric_scores": {},
  "concept_coverage": {},
  "missing_concepts": [],
  "weighted_score": 0.0,
  "summary": ""
}

def _build_numeric_prompt(question: str, answer: str, ideal: str, rubric_json: Dict[str, Any]) -> str:
    return f"""You are an evaluator scoring a candidate's interview answer.
Return JSON only with keys: {list(NUMERIC_SCHEMA.keys())}.

Question: {question}
Answer: {answer}
Ideal Answer: {ideal or "N/A"}
Rubric: {json.dumps(rubric_json)}
"""

def call_openai(prompt: str, temperature: float = 0.2) -> str:
    if os.getenv("MOCK_MODE","0") == "1":
        # deterministic mock output
        return ("{\"rubric_scores\": {\"technical_correctness\": 7, \"depth_of_reasoning\": 6}, "
                "\"concept_coverage\": {}, \"missing_concepts\": [], \"weighted_score\": 6.8, "
                "\"summary\": \"Solid but missing metrics.\"}")
    client = _client_openai()
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}], temperature=temperature
    )
    return resp.choices[0].message.content

def evaluate_answer(qid: str, question: str, answer: str, rubric_json: Dict[str, Any], ideal_answer: str="") -> Dict[str, Any]:
    prompt = _build_numeric_prompt(question, answer, ideal_answer, rubric_json)
    raw = call_openai(prompt, 0.1)
    try:
        data = json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s:e+1]) if s!=-1 and e!=-1 and e>s else {
            "rubric_scores": {}, "concept_coverage": {}, "missing_concepts": [], "weighted_score": 0.0, "summary": "parse_failed"
        }
    data["question_id"] = qid
    return data
