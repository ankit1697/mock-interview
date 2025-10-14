import openai
import time
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

resume_path = "resume.pdf"
jd_path = "jd.txt"
resume = read_file(resume_path)
job_description = read_file(jd_path)

# Prepare messages for the initial question
messages = [
    {"role": "system", "content": "You are an expert AI interviewer for data science roles."},
    {"role": "user", "content": f"""
Given the resume and job description below, generate 1 thoughtful and role-specific interview question.

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

initial_question = response.choices[0].message.content
print("\nInitial Question:")
print(initial_question)
print(f"Took {round(end - start, 2)} seconds\n")

# Store messages to continue conversation
messages.append({"role": "assistant", "content": initial_question})

# Function to generate side-by-side follow-up questions (only created once after the first question)
def generate_side_by_side_questions():
    messages.append({"role": "user", "content": f"""
Based on the initial interview question generated above, generate 3 additional follow-up questions that are related to those questions.
"""})
    
    start = time.time()
    followup_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()
    
    side_by_side_questions = followup_response.choices[0].message.content
    return side_by_side_questions.strip().split('\n')

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
    messages.append({"role": "system", "content": f"""
Provide the most logical follow-up question based on the user response: "{closest_question}"
"""})

    start = time.time()
    closest_answer_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()

    closest_answer = closest_answer_response.choices[0].message.content
    print(f"\nFollow-Up Answer: {closest_answer}")
    print(f"Took {round(end - start, 2)} seconds\n")

    messages.append({"role": "assistant", "content": closest_answer})