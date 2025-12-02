import openai
import time
import re
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
from utils import read_file
from difflib import SequenceMatcher

load_dotenv(find_dotenv(), override=True)
key = os.getenv("OPENAI_API_KEY")
key = (key.strip() if key else None)
print("OPENAI_API_KEY endswith:", (key[-4:] if key else "None"))
if key and not key.startswith("sk-"):
    print("Warning: OPENAI_API_KEY does not start with 'sk-'")
client = OpenAI(api_key=key)

def clean_response(text):
    """Remove formatting symbols and unnecessary characters to make responses look natural"""
    if not text:
        return text
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
    text = re.sub(r'__(.*?)__', r'\1', text)  # Remove underline
    text = re.sub(r'_(.*?)_', r'\1', text)  # Remove italic underscore
    
    # Remove list markers and numbering
    text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)  # Remove numbered lists
    text = re.sub(r'^[-•*]\s*', '', text, flags=re.MULTILINE)  # Remove bullet points
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    
    # Remove common AI formatting patterns
    text = re.sub(r'^Here\'s.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^Let me.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^I\'d like to.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove quotes around the entire response if present
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    return text.strip()

resume_path = "resume.pdf"
jd_path = "jd.txt"
resume = read_file(resume_path)
job_description = read_file(jd_path)

# Prepare messages for the initial question
messages = [
    {"role": "system", "content": "You are a professional interviewer conducting a data science interview. Speak naturally and conversationally, as if you're having a real conversation. Do not use formatting, bullet points, numbered lists, or any special characters. Just ask questions in plain, natural language."},
    {"role": "user", "content": f"""
Based on this resume and job description, ask one natural interview question. Write it as if you're speaking directly to the candidate in a conversational tone. No formatting, no lists, just a plain question.

Resume:
{resume}

Job Description:
{job_description}
"""}
]

start = time.time()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
end = time.time()

initial_question = clean_response(response.choices[0].message.content)
print("\nInitial Question:")
print(initial_question)
print(f"Took {round(end - start, 2)} seconds\n")

# Store messages to continue conversation
messages.append({"role": "assistant", "content": initial_question})

# Function to generate side-by-side follow-up questions (only created once after the first question)
def generate_side_by_side_questions():
    messages.append({"role": "user", "content": f"""
Based on the initial interview question, generate 3 additional natural follow-up questions. Write each question on a separate line, in plain conversational language. No formatting, no numbering, no bullet points, just plain questions.
"""})
    
    start = time.time()
    followup_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()
    
    raw_questions = clean_response(followup_response.choices[0].message.content)
    # Split by newlines and clean each question
    side_by_side_questions = [clean_response(q.strip()) for q in raw_questions.split('\n') if q.strip()]
    return side_by_side_questions

# Initialize the side-by-side questions after the first question is asked
side_by_side_questions = generate_side_by_side_questions()

# Function to find the closest match from the side-by-side questions based on user input
def get_closest_match(input_text, question_list):
    similarities = [(question, SequenceMatcher(None, input_text, question).ratio()) for question in question_list]
    closest_match = max(similarities, key=lambda x: x[1])
    return closest_match[0]

# Loop for dynamic follow-up after the user's response
while True:
    user_input = input("Your Answer (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    # Compare the user's response to the side-by-side questions and get the closest match
    closest_question = get_closest_match(user_input, side_by_side_questions)

    # Ask the most relevant follow-up question based on the user's response
    messages.append({"role": "user", "content": f"""
Based on the candidate's response, ask a natural follow-up question. Write it as if you're speaking directly to them in a conversational tone. No formatting, no lists, just a plain question that flows naturally from their answer.
"""})

    start = time.time()
    closest_answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()

    closest_answer = clean_response(closest_answer_response.choices[0].message.content)
    print(f"\nFollow-Up Question: {closest_answer}")
    print(f"Took {round(end - start, 2)} seconds\n")

    messages.append({"role": "assistant", "content": closest_answer})