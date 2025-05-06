import os
import json
import time
import openai
from openai import OpenAI
from PyPDF2 import PdfReader

openai.api_key = os.getenv("OPENAI_API_KEY")


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
        client = OpenAI()

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
            client = OpenAI()

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


def generate_initial_question(structured_resume, structured_jd):
    """Generate the initial interview question based on structured resume and job description"""
    # Prepare resume and job description in a structured format
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
        "company": structured_jd.get("company", ""),
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
Given the structured resume and job description below, generate 1 thoughtful and role-specific technical interview question.

STRUCTURED RESUME:
{resume_json}

STRUCTURED JOB DESCRIPTION:
{jd_json}

Guidelines for your question:
1. Be specific to the candidate's background and the job requirements
2. Focus on technical knowledge and practical application
3. Target skills or experiences that appear most relevant for this role
4. Keep the question under 30 words and end with a question mark
5. Make it conversational but substantive
"""}
    ]

    # Call OpenAI API
    client = OpenAI()

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


def generate_dynamic_question(messages, user_response, personal_profile=None):
    """Generate a dynamic follow-up question based on the user's response"""
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

    system_message = {
        "role": "system",
        "content": f"""
            You are an expert technical interviewer for data science roles.
            {profile_context}
            Generate a thoughtful follow-up question based on the candidate's response.
            The question should:
            1. Dig deeper into a specific aspect mentioned by the candidate
            2. Be conversational but technically substantive
            3. Be no longer than 30 words
            4. End with a question mark
        """
    }

    # Create a new messages array with the system message first
    formatted_messages = [system_message] + messages

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
        initial_question, self.messages = generate_initial_question(self.structured_resume, self.structured_jd)
        self.current_question = initial_question
        self.interview_data.append({"question": initial_question, "answer": None, "evaluation": None})
        self.total_questions_asked += 1
        return f"{greeting}\n\n{initial_question}"

    def is_clarification_request(self, user_response):
        clarification_prompt = f"""
        Determine if the candidate is asking for clarification or rephrasing of the question instead of providing an actual answer. Respond only 'yes' or 'no'.

        Candidate's Response: {user_response}
        """
        check_response = OpenAI().chat.completions.create(
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

        response = OpenAI().chat.completions.create(
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
        response = OpenAI().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI interviewer and evaluator for data science roles."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def add_answer(self, answer):
        if self.is_clarification_request(answer):
            clarification_response = self.rephrase_current_question()
            return clarification_response, "Question clarified."

        self.interview_data[self.current_question_idx]["answer"] = answer

        # Evaluate immediately
        evaluation_feedback = self.evaluate_answer(answer, self.current_question)
        self.interview_data[self.current_question_idx]["evaluation"] = evaluation_feedback

        # 🛠 FIX: Move the pointer forward after saving answer
        self.current_question_idx += 1

        return evaluation_feedback, "Answer recorded and evaluated."


    def should_ask_follow_up(self, user_response):
        follow_up_check_prompt = f"Was the candidate's answer complete and clear? Answer with 'yes' or 'no'.\nCandidate's answer: {user_response}"
        check_response = OpenAI().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": follow_up_check_prompt}],
            temperature=0
        )
        return 'no' in check_response.choices[0].message.content.lower()

    def next_question(self):
        if self.total_questions_asked >= 5:
            return "We've reached the end of your interview. Thank you for your time!"

        # Add last assistant/user interaction to memory
        if self.current_question_idx > 0:
            last_q = self.interview_data[self.current_question_idx - 1]
            if last_q["answer"]:
                self.messages.append({"role": "assistant", "content": last_q["question"]})
                self.messages.append({"role": "user", "content": last_q["answer"]})

        if self.follow_up_count < 3 and self.should_ask_follow_up(self.interview_data[self.current_question_idx - 1]["answer"]):
            next_q = generate_dynamic_question(self.messages.copy(), self.interview_data[self.current_question_idx - 1]["answer"], self.personal_profile)
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

        summary_response = OpenAI().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI interviewer and evaluator for data science roles."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        return summary_response.choices[0].message.content.strip()
