import openai
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils import read_file

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resume_path = "resume.pdf"
jd_path = "jd.txt"
resume = read_file(resume_path)
job_description = read_file(jd_path)

messages = [
    {"role": "system", "content": "You are an expert AI interviewer for data science roles."},
    {"role": "user", "content": f"""
Given the resume and job description below, generate 5 thoughtful and role-specific interview questions.

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

initial_questions = response.choices[0].message.content
print("\n Initial Questions:")
print(initial_questions)
print(f" Took {round(end - start, 2)} seconds\n")

messages.append({"role": "assistant", "content": initial_questions})

while True:
    user_input = input(" Your Answer (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "system", "content": "Ask one thoughtful follow-up question based on the user's answer."})

    start = time.time()
    followup_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()

    followup_question = followup_response.choices[0].message.content
    print(f"\n Follow-up Question: {followup_question}")
    print(f" Took {round(end - start, 2)} seconds\n")

    messages.append({"role": "assistant", "content": followup_question})
