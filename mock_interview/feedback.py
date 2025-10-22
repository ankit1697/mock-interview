import json

def parse_feedback_sections(text: str) -> dict:
    blk = {"strengths": "", "areas_to_improve": "", "suggestions": ""}
    cur = None
    for line in text.splitlines():
        l = line.strip()
        if l.startswith("✅"): cur = "strengths"; blk[cur] = ""
        elif l.startswith("⚙️"): cur = "areas_to_improve"; blk[cur] = ""
        elif l.startswith("💡"): cur = "suggestions"; blk[cur] = ""
        elif cur: blk[cur] += (("\n" if blk[cur] else "") + l)
    return blk

def build_final_feedback(question: str, answer: str, evaluation_json: dict, call_openai) -> str:
    prompt = f"""You are an interview feedback coach.
Format exactly:
✅ Strengths
⚙️ Areas to Improve
💡 Suggestions

Question: {question}
Answer: {answer}
Evaluation JSON: {json.dumps(evaluation_json)}
"""
    return call_openai(prompt, temperature=0.2).strip()
