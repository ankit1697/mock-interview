import os
import json
import time
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
import requests

_dotenv_path = find_dotenv()
load_dotenv(_dotenv_path, override=True)

# Normalize and validate OpenAI key
_openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = _openai_key
openai_client = OpenAI(api_key=_openai_key)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def read_file(file_path):
    """Generic function to read file content based on extension"""
    try:
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()

        # Read the file content based on type
        if file_extension == '.pdf':
            return read_pdf(file_path)
        elif file_extension == '.docx':
            return read_docx(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension in ['.csv', '.xlsx', '.xls']:
            # For tabular formats, convert to text representation
            return read_tabular(file_path)
        else:
            # Try to read as text for unknown file types
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return f"Could not read file {file_path}"


def read_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""


def read_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return ""


def parse_resume_with_ai(resume_text):
    """Use AI to extract structured information from resume text"""
    try:
        client = openai_client

        prompt = f"""
        Extract structured information from the following resume:

        {resume_text}

        Return the information in this JSON format:
        {{
            "contact_info": {{
                "name": "",
                "email": "",
                "phone": "",
                "location": ""
            }},
            "summary": "",
            "education": [
                {{
                    "degree": "",
                    "institution": "",
                    "dates": "",
                    "gpa": "",
                    "details": []
                }}
            ],
            "experience": [
                {{
                    "title": "",
                    "company": "",
                    "dates": "",
                    "location": "",
                    "responsibilities": []
                }}
            ],
            "skills": {{
                "technical": [],
                "soft": [],
                "languages": [],
                "tools": []
            }},
            "projects": [
                {{
                    "name": "",
                    "description": "",
                    "technologies": [],
                    "outcomes": []
                }}
            ],
            "certifications": [],
            "years_of_experience": 0,
            "domain_expertise": []
        }}

        If a field can't be determined from the resume, use null or an empty array as appropriate.
        For "years_of_experience", determine the total years of professional experience based on work history.
        For "domain_expertise", identify industry domains where the candidate has experience.
        """

        messages = [
            {"role": "system", "content": "You are an expert resume parser. Extract structured information from resumes accurately."},
            {"role": "user", "content": prompt}
        ]

        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        end = time.time()

        structured_resume = json.loads(response.choices[0].message.content)
        print(f"Resume parsed in {round(end - start, 2)} seconds")

        return structured_resume

    except Exception as e:
        print(f"Error parsing resume with AI: {str(e)}")
        return {
            "contact_info": {},
            "education": [],
            "experience": [],
            "skills": {},
            "years_of_experience": 0,
            "domain_expertise": []
        }

def extract_resume_structure(resume_path):
    """Extract structured information from a resume file"""
    try:
        resume_text = read_file(resume_path)
        print(f"Successfully read resume from {resume_path}")

        # Use OpenAI to extract structured information
        structured_resume = parse_resume_with_ai(resume_text)
        return structured_resume

    except Exception as e:
        print(f"Error extracting resume structure: {str(e)}")
        # Return a minimal structure if parsing fails
        return {
            "contact_info": {},
            "education": [],
            "experience": [],
            "skills": [],
            "full_text": "Could not parse resume"
        }


def extract_job_description_structure_from_text(jd_text):
        try:
            client = openai_client

            prompt = f"""
            Extract structured information from the following job description:

            {jd_text}

            Return the information in this JSON format:
            {{
                "role": "",
                "company": "",
                "location": "",
                "skills_required": [],
                "summary": "",
                "responsibilities": [],
                "qualifications": [],
                "years_of_experience_required": 0
            }}

            If any information is missing, leave it as null or an empty list.
            """

            messages = [
                {"role": "system", "content": "You are an expert in parsing job descriptions accurately."},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )

            structured_jd = json.loads(response.choices[0].message.content)
            return structured_jd

        except Exception as e:
            print(f"Error parsing JD from text: {e}")
            return {
                "role": None,
                "company": None,
                "location": None,
                "skills_required": [],
                "summary": "",
                "responsibilities": [],
                "qualifications": [],
                "years_of_experience_required": 0
            }


def create_personal_profile(structured_resume, structured_jd):
    """Create a mini profile from parsed resume and job description."""
    try:
        profile = {
            "name": None,
            "role": None,
            "company": None
        }

        if structured_resume:
            profile["name"] = structured_resume.get("contact_info", {}).get("name")

        if structured_jd:
            profile["role"] = structured_jd.get("role")
            profile["company"] = structured_jd.get("company")

        return profile
    except Exception as e:
        print(f"Error creating personal profile: {e}")
        return {
            "name": None,
            "role": None,
            "company": None
        }


def get_recent_interview_questions(company_name):
    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are a helpful AI assistant that conducts mock interviews.
Rules:
1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
2. Do not show the intermediate steps information.
Steps:
1. Decide if the answer should be a brief sentence or a list of suggestions.
2. If it is a list of suggestions, first, write a brief and natural introduction based on the original query.
3. Followed by a list of suggestions, each suggestion should be split by two newlines.
Question : Give me top 5 recent non coding technical questions for data science interviews at {company_name}. If the company is too small and you really dont find anything, give a list of generic questions"""
    
    data = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error getting recent interview questions: {e}")
        return f"Unable to fetch recent questions for {company_name}. Proceeding with standard interview."



def generate_initial_question(structured_resume, structured_jd, company=""):
    """Generate the initial interview question based on structured resume and job description"""
    # Prepare resume and job description in a structured format

    recent_questions = get_recent_interview_questions(company)

    resume_summary = {
        "name": structured_resume.get("contact_info", {}).get("name", ""),
        "experience": [
            {
                "title": exp.get("title", ""),
                "company": exp.get("company", ""),
                "responsibilities": exp.get("responsibilities", [])[:3]
            } for exp in structured_resume.get("experience", [])[:3]
        ],
        "education": [
            {
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", "")
            } for edu in structured_resume.get("education", [])
        ],
        "skills": structured_resume.get("skills", {}),
        "projects": [proj.get("name", "") for proj in structured_resume.get("projects") or []][:3]
    }

    jd_summary = {
        "title": structured_jd.get("title", ""),
        "company": company,
        "responsibilities": structured_jd.get("responsibilities", [])[:5],  # Limit to 5 key responsibilities
        "required_skills": structured_jd.get("requirements", {}).get("required_skills", []),
        "industry": structured_jd.get("industry", "")
    }

    # Convert to JSON strings
    resume_json = json.dumps(resume_summary, indent=2)
    jd_json = json.dumps(jd_summary, indent=2)

    # Prepare messages
    messages = [
        {"role": "system", "content": "You are an expert AI interviewer for data science roles."},
        {"role": "user", "content": f"""
Given the structured resume and job description below, generate a thoughtful and role-specific technical interview question.

STRUCTURED RESUME:
{resume_json}

STRUCTURED JOB DESCRIPTION:
{jd_json}

RECENT INTERVIEW QUESTIONS FROM THE COMPANY:
{recent_questions}

Guidelines for your question:
1. You can use the list of company-wise question above or the candidate's background (only if relevant to the position) and the job requirements.
2. Focus on technical knowledge and practical application
3. Target skills or experiences that appear most relevant for this role
4. Keep the question under 30 words and end with a question mark
5. Make it conversational but substantive
6. Ask only one question at a time
"""}
    ]

    print(recent_questions)

    # Call OpenAI API
    client = openai_client

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    end = time.time()

    initial_question = response.choices[0].message.content
    print(f"\nInitial Question:\n{initial_question}")
    print(f"Took {round(end - start, 2)} seconds\n")

    return initial_question, messages


def generate_dynamic_question(messages, user_response, personal_profile=None, interview_type="general"):
    """Generate a dynamic follow-up question based on the user's response and interview type"""
    # Add the user's response to the conversation
    messages.append({"role": "user", "content": user_response})

    # Fall back to OpenAI directly
    print("Generating dynamic follow-up question")

    client = OpenAI()

    # Create a new system message with profile context if available
    profile_context = ""
    if personal_profile:
        profile_context = f"""
        Candidate Profile:
        - Name: {personal_profile.get('name', 'Candidate')}
        - Role applying for: {personal_profile.get('role', 'Data Science Role')}
        - Experience level: {personal_profile.get('experience', 'Unknown')} years
        - Technical skills: {', '.join(personal_profile.get('skills', {}).get('technical', [])[:5])}
        - Background: {', '.join(personal_profile.get('domain_expertise', []))}
        """

    # Base instruction for all interview types
    base_instruction = """
        Generate a thoughtful follow-up question based on the candidate's response.
        The question can:
        1. Be no longer than 30 words
        2. End with a question mark
        3. Be conversational but technically substantive
        4. Dig deeper into a specific aspect mentioned by the candidate, but not necessarily!

    """

    # Add technical interview specific instruction if needed
    if interview_type.lower() == "technical":
        base_instruction += """
        5. Ask precise data science technical questions that have a specific answer that can be evaluated (ex: explain a concept...)
        6. Don't hesitate to ask questions that are completely unrelated with the previous conversation (you can ask like boosting vs bagging or any other common data science question with a fixed answer)
        7. Ask no more than 1 question which is not a data science classic interview question that could be asked regardless of the background
        """

    system_message = {
        "role": "system",
        "content": f"""
            You are an expert technical interviewer for data science roles.
            {profile_context}
            {base_instruction}
        """
    }

    # Create a new messages array with the system message first
    formatted_messages = [system_message] + messages

    print(base_instruction)

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=formatted_messages,
        temperature=0.2
    )
    end = time.time()

    follow_up_question = response.choices[0].message.content
    print(f"\nDynamic Follow-Up Question: {follow_up_question}")
    print(f"Took {round(end - start, 2)} seconds\n")

    return follow_up_question


class InterviewSession:
    def __init__(self, resume_obj, job_description_text, company=None):
        self.resume_obj = resume_obj  # Save the Resume object
        self.job_description_text = job_description_text or "N/A"
        self.company = company or "N/A"

        self.structured_resume = {
            "contact_info": {
                "name": resume_obj.name,
                "email": resume_obj.email,
                "phone": resume_obj.phone,
                "location": resume_obj.location
            },
            "summary": resume_obj.summary,
            "education": resume_obj.education,
            "experience": resume_obj.experience,
            "skills": resume_obj.skills,
            "projects": resume_obj.projects,
            "certifications": resume_obj.certifications,
            "domain_expertise": resume_obj.domain_expertise,
            "years_of_experience": resume_obj.years_of_experience
        }

        self.structured_jd = extract_job_description_structure_from_text(self.job_description_text)
        self.personal_profile = create_personal_profile(self.structured_resume, self.structured_jd)

        self.interview_data = []
        self.current_question_idx = 0
        self.follow_up_count = 0
        self.total_questions_asked = 0
        self.messages = []
        self.current_question = None



    def start_interview(self):
        greeting = f"Hi {self.personal_profile.get('name', 'there')}, nice to meet you! Let's get started with your interview for the {self.personal_profile.get('role', 'data science')} role at {self.company}"
        initial_question, self.messages = generate_initial_question(self.structured_resume, self.structured_jd, self.company)
        self.current_question = initial_question
        self.interview_data.append({"question": initial_question, "answer": None, "evaluation": None})
        self.total_questions_asked += 1
        return f"{greeting}\n\n{initial_question}"

    def is_clarification_request(self, user_response):
        clarification_prompt = f"""
        Determine if the candidate is asking for clarification or rephrasing of the question instead of providing an actual answer. Respond only 'yes' or 'no'.

        Candidate's Response: {user_response}
        """
        check_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": clarification_prompt}],
            temperature=0
        )
        return 'yes' in check_response.choices[0].message.content.lower()

    def rephrase_current_question(self):
        rephrase_prompt = f"""
        You are an AI assistant helping candidates understand interview questions.

        Task:
        - Rephrase the following question to make it simpler and easier to understand.
        - Use shorter sentences if needed.
        - If the question involves multiple parts, break it into bullet points.
        - Optionally, add a tiny 1-line hint if you think the candidate may still struggle.

        Important: Always rephrase. Even if the question is already clear, still rewrite it in a different way.

        Original Question:
        {self.current_question}
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": rephrase_prompt}],
            temperature=0.5  # slight randomness to force rephrasing
        )
        self.current_question = response.choices[0].message.content.strip()
        return self.current_question


    def evaluate_answer(self, answer, question=None):
        evaluation_prompt = f"""
        You are an expert interviewer and evaluator for data science roles. Evaluate the following answer:

        Question: {question if question else self.current_question}
        Candidate's Answer: {answer}

        Evaluate the candidate's answer on the following criteria:
        1. Technical Accuracy (0-10)
        2. Relevance (0-10)
        3. Depth (0-10)
        4. Communication (0-10)
        5. Practical Application (0-10)

        For each criterion, provide a score, a short feedback, and an improvement suggestion.
        Then provide an Overall Score (0-100) with a 2-3 sentence summary.
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI interviewer and evaluator for data science roles."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def handle_unknown_answer(self, answer):
        """When candidate says they don't know, produce a short hint and a simpler follow-up question."""
        prompt = f"""
You are an empathetic technical interviewer helping a candidate who says they don't know the answer.
Task: Provide a concise, 1-2 sentence hint that guides the candidate how to think about the problem (no spoilers or full solution), then provide one simpler follow-up question they can attempt. Format exactly as:
HINT: <hint text>
FOLLOWUP: <follow-up question?>

Candidate's raw reply: {answer}
"""

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            text = resp.choices[0].message.content.strip()
            # Extract lines
            hint = None
            follow = None
            for line in text.splitlines():
                if line.strip().lower().startswith('hint:'):
                    hint = line.split(':', 1)[1].strip()
                if line.strip().lower().startswith('followup:') or line.strip().lower().startswith('follow-up:'):
                    follow = line.split(':', 1)[1].strip()

            if not hint:
                # fallback: first paragraph
                hint = text.split('\n\n')[0].strip()
            if not follow:
                # fallback: last line
                follow = text.splitlines()[-1].strip()

            return hint, follow
        except Exception as e:
            print(f"Error generating hint/followup: {e}")
            return ("Try breaking the problem into smaller steps.", "Can you describe the first step you would take?")

    def add_answer(self, answer):
        if self.is_clarification_request(answer):
            clarification_response = self.rephrase_current_question()
            return clarification_response, "Question clarified."

        # Save the raw answer
        self.interview_data[self.current_question_idx]["answer"] = answer

        # Add the assistant question and user's answer to the conversation messages so follow-up generation sees them
        try:
            self.messages.append({"role": "assistant", "content": self.current_question})
            self.messages.append({"role": "user", "content": answer})
        except Exception:
            pass

        # Detect explicit 'I don't know' type responses (simple heuristics)
        low = answer.strip().lower()
        unknown_indicators = ["don't know", "dont know", "idk", "not sure", "no idea", "i'm not sure", "i am not sure"]
        if any(ind in low for ind in unknown_indicators):
            # Provide a hint and a simpler follow-up question instead of marking this answer final
            hint, follow = self.handle_unknown_answer(answer)
            # Store a brief evaluation note
            self.interview_data[self.current_question_idx]["evaluation"] = f"Candidate indicated they didn't know. Provided hint: {hint}"

            # Set follow-up as the current question and append a slot
            self.current_question = follow
            self.interview_data.append({"question": follow, "answer": None, "evaluation": None})
            self.total_questions_asked += 1

            # Do not increment current_question_idx so the original question remains accessible
            return hint, "Hint provided, follow-up question asked."

        # Otherwise: Evaluate normally
        evaluation_feedback = self.evaluate_answer(answer, self.current_question)
        self.interview_data[self.current_question_idx]["evaluation"] = evaluation_feedback

        # Move pointer forward after saving answer
        self.current_question_idx += 1

        return evaluation_feedback, "Answer recorded and evaluated."


    def should_ask_follow_up(self, user_response):
        follow_up_check_prompt = f"Was the candidate's answer complete and clear? Answer with 'yes' or 'no'.\nCandidate's answer: {user_response}"
        check_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": follow_up_check_prompt}],
            temperature=0
        )
        return 'no' in check_response.choices[0].message.content.lower()

    def next_question(self, interview_type):
        if self.total_questions_asked >= 5:
            return "We've reached the end of your interview. Thank you for your time!"

        # Add last assistant/user interaction to memory
        if self.current_question_idx > 0:
            last_q = self.interview_data[self.current_question_idx - 1]
            if last_q["answer"]:
                self.messages.append({"role": "assistant", "content": last_q["question"]})
                self.messages.append({"role": "user", "content": last_q["answer"]})

        if self.follow_up_count < 3 and self.should_ask_follow_up(self.interview_data[self.current_question_idx - 1]["answer"]):
            next_q = generate_dynamic_question(self.messages.copy(), self.interview_data[self.current_question_idx - 1]["answer"], self.personal_profile, interview_type)
            self.follow_up_count += 1
        else:
            next_q, _ = generate_initial_question(self.structured_resume, self.structured_jd)
            self.follow_up_count = 0

        # Set current question
        self.current_question = next_q

        # Append new question slot to interview_data
        self.interview_data.append({"question": next_q, "answer": None, "evaluation": None})

        self.total_questions_asked += 1
        self.messages = self.messages[-7:]  # Keep context short

        return next_q



    def generate_per_question_feedback(self):
        feedbacks = []
        for idx, entry in enumerate(self.interview_data):
            if entry["answer"] and entry["evaluation"]:
                feedbacks.append(f"Feedback for Question {idx+1}:\n{entry['evaluation']}\n")
        return "\n".join(feedbacks)

    def get_interview_summary(self):
        # No need to re-evaluate here
        transcript = ""
        for i, q_data in enumerate(self.interview_data):
            transcript += f"Q{i+1}: {q_data['question']}\nA{i+1}: {q_data['answer']}\nEvaluation: {q_data['evaluation']}\n---\n"

        summary_prompt = f"""
        You are an expert interviewer for data science roles. Analyze the following interview and provide:
        1. Overall Assessment (2-3 paragraphs)
        2. Key Strengths (3-5 bullet points)
        3. Areas for Improvement (2-3 bullet points)
        4. Technical Competency (1-10) with explanation
        5. Communication Skills (1-10) with explanation

        Interview:
        {transcript}
        """

        summary_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI interviewer and evaluator for data science roles."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        return summary_response.choices[0].message.content.strip()
