# evaluator.py
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RUBRIC = {
    "Technical Skill": 0.4,
    "Problem-Solving": 0.3,
    "Communication": 0.2,
    "Impact/Results": 0.1
}

def evaluate_transcript(transcript):
    """Evaluate a saved transcript using STAR + Google rubric."""
    # Format transcript for evaluation
    conversation = "\n".join([f"{speaker}: {text}" for speaker, text in transcript])
    
    prompt = f"""
    Evaluate this interview transcript based on Google's hiring rubric and STAR framework:
    {conversation}

    Scoring Guide:
    1. **Technical Skill** (Weight: {RUBRIC["Technical Skill"]*100}%):
       - Tools/Methods mentioned? Depth of technical detail?
    2. **Problem-Solving** (Weight: {RUBRIC["Problem-Solving"]*100}%):
       - Logical reasoning? Approach to challenges?
    3. **Communication** (Weight: {RUBRIC["Communication"]*100}%):
       - Clarity? Structure (STAR adherence)?
    4. **Impact/Results** (Weight: {RUBRIC["Impact/Results"]*100}%):
       - Quantifiable outcomes? Business impact?

    Provide:
    1. Scores (0-5) per category.
    2. Weighted total score (0-5).
    3. Feedback for improvement.
    """
    
    evaluation = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    return evaluation.choices[0].message.content

# Example Django view (pseudo-code)
# from django.http import JsonResponse
# def evaluate_interview(request):
#     transcript = request.POST.get("transcript")  # Get from DB or frontend
#     evaluation = evaluate_transcript(transcript)
#     return JsonResponse({"evaluation": evaluation})


if __name__ == "__main__":
    # Simulate a saved transcript
    test_transcript = [
        ("AI", "Describe a data science project."),
        ("Candidate", "I built a model with Python."),
        ("AI", "Which libraries did you use?")
    ]
    print(evaluate_transcript(test_transcript))