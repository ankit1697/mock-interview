"""
Interview session management using QSE (Question Selection Engine) and resume_parser.
All resume parsing is handled by resume_parser.py, and all question generation/orchestration
is handled by question_selection_engine.py.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from .evaluation_engine import (
    evaluate_interview_json,
    format_answer_evaluation_feedback,
    format_interview_summary
)

# Load environment variables first
_dotenv_path = find_dotenv()
load_dotenv(_dotenv_path, override=True)

# Normalize and validate OpenAI key
_openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=_openai_key)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Add cgpt directory to path for QSE and resume_parser imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CGPT_PATH = PROJECT_ROOT / 'cgpt'
if str(CGPT_PATH) not in sys.path:
    sys.path.insert(0, str(CGPT_PATH))

# Import resume_parser (if available)
RESUME_PARSER_AVAILABLE = False
try:
    from resume_parser import parse_resume_from_file
    RESUME_PARSER_AVAILABLE = True
    # Patch resume_parser's OpenAI client
    import resume_parser as rp_module
    if hasattr(rp_module, 'client'):
        rp_module.client = openai_client
except ImportError as e:
    print(f"Warning: resume_parser not available: {e}")
    RESUME_PARSER_AVAILABLE = False

# Import QSE functions (patch OpenAI client to use our key)
QSE_AVAILABLE = False
try:
    import question_selection_engine as qse_module
    # Patch the QSE's OpenAI client to use our key BEFORE importing functions
    if hasattr(qse_module, 'client'):
        qse_module.client = openai_client
    else:
        qse_module.client = openai_client
    
    # Set small talk turns to allow conversational flow (3 turns = 3 conversational exchanges)
    if hasattr(qse_module, 'MAX_SMALLTALK_TURNS'):
        qse_module.MAX_SMALLTALK_TURNS = 3
    
    # Now import the functions (they'll use the patched client)
    from question_selection_engine import (
        build_scored_question_pool,
        start_smalltalk,
        continue_smalltalk,
        get_first_interview_question,
        classify_answer_behavior,
        clarify_question,
        needs_followup,
        generate_followup_question,
        get_next_diverse_question,
        normalize_question,
        transition_phrase,
    )
    QSE_AVAILABLE = True
    print("[script.py] ✅ QSE module loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import QSE functions: {e}")
    import traceback
    traceback.print_exc()
    QSE_AVAILABLE = False
    # Define fallback functions
    def build_scored_question_pool(*args, **kwargs):
        return []
    def start_smalltalk(*args, **kwargs):
        return ("Hi! Let's get started.", [], 1)
    def continue_smalltalk(*args, **kwargs):
        return ("Thanks! Let's begin the interview.", [], 1, True, False, 0)
    def get_first_interview_question(*args, **kwargs):
        return "Tell me about yourself and your experience."
    def classify_answer_behavior(*args, **kwargs):
        return "on_topic"
    def clarify_question(question):
        return question
    def needs_followup(*args, **kwargs):
        return False
    def generate_followup_question(*args, **kwargs):
        return ""
    def get_next_diverse_question(*args, **kwargs):
        return None
    def normalize_question(q):
        return q if isinstance(q, str) else (q.get("question", "") if isinstance(q, dict) else str(q))
    def transition_phrase(context="next_topic"):
        if context == "next_topic":
            return "That's alright! Let's move on to something else."
        return "Alright, let's continue."


def convert_resume_to_qse_format(resume_obj) -> Dict[str, Any]:
    """
    Convert Django Resume model format to QSE expected format.
    First tries to use resume_parser if file is available, otherwise converts from DB fields.
    QSE expects: Name, Professional Summary, List of Core skills, Skills, 
    List of Project domains, Academic Projects, List of Work context
    """
    # Try to use resume_parser if available and file exists
    if RESUME_PARSER_AVAILABLE and hasattr(resume_obj, 'file') and resume_obj.file:
        try:
            file_path = resume_obj.file.path
            if os.path.exists(file_path):
                print(f"[convert_resume_to_qse_format] Using resume_parser on file: {file_path}")
                parsed = parse_resume_from_file(file_path)
                # Ensure all expected keys exist with defaults (using new model field names)
                parsed.setdefault("Name", resume_obj.name or "")
                parsed.setdefault("Professional Summary", resume_obj.summary or "")
                parsed.setdefault("List of Core skills", resume_obj.core_skills or parsed.get("Skills", []))
                parsed.setdefault("Skills", parsed.get("List of Core skills", []))
                parsed.setdefault("List of Project domains", resume_obj.project_domains or [])
                parsed.setdefault("Academic Projects", resume_obj.academic_projects or parsed.get("Academic Projects", []))
                parsed.setdefault("List of Work context", resume_obj.work_context or [])
                parsed.setdefault("Education", resume_obj.education or [])
                parsed.setdefault("Work Experience", resume_obj.work_experience or [])
                parsed.setdefault("Professional Projects", resume_obj.professional_projects or [])
                parsed.setdefault("Certifications", resume_obj.certifications or [])
                parsed.setdefault("Total Experience", resume_obj.total_experience or 0)
                
                # Ensure Skills is a list if it's a dict
                if isinstance(parsed.get("Skills"), dict):
                    tech_skills = parsed["Skills"].get("technical", [])
                    parsed["Skills"] = tech_skills
                    if not parsed.get("List of Core skills"):
                        parsed["List of Core skills"] = tech_skills[:10]
                
                print(f"[convert_resume_to_qse_format] Parsed resume keys: {list(parsed.keys())}")
                print(f"[convert_resume_to_qse_format] Core skills count: {len(parsed.get('List of Core skills', []))}")
                return parsed
        except Exception as e:
            print(f"[convert_resume_to_qse_format] Error using resume_parser, falling back to converter: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: Convert from Django model fields (using new field names)
    # Extract skills
    skills_data = resume_obj.skills or {}
    technical_skills = skills_data.get("technical", []) if isinstance(skills_data, dict) else []
    all_skills = technical_skills if technical_skills else (skills_data if isinstance(skills_data, list) else [])
    # Prefer core_skills field if available, otherwise use extracted skills
    if resume_obj.core_skills:
        core_skills_list = resume_obj.core_skills if isinstance(resume_obj.core_skills, list) else []
        all_skills = core_skills_list if core_skills_list else all_skills
    
    # Extract projects (use new professional_projects field)
    projects_list = resume_obj.professional_projects or []
    project_domains = resume_obj.project_domains or []
    project_descriptions = []
    if isinstance(projects_list, list):
        for proj in projects_list:
            if isinstance(proj, dict):
                project_descriptions.append(proj.get("description", ""))
                if not project_domains:
                    project_domains.append(proj.get("name", ""))
    
    # Extract experience/work context (use new work_experience and work_context fields)
    experience_list = resume_obj.work_experience or []
    work_context = resume_obj.work_context or []
    if not work_context and isinstance(experience_list, list):
        for exp in experience_list:
            if isinstance(exp, dict):
                work_context.append(f"{exp.get('title', '')} at {exp.get('company', '')}")
    
    # Build QSE format
    qse_resume = {
        "Name": resume_obj.name or "",
        "Professional Summary": resume_obj.summary or "",
        "List of Core skills": all_skills[:10] if isinstance(all_skills, list) else [],  # Top 10 skills
        "Skills": all_skills if isinstance(all_skills, list) else [],  # All skills
        "List of Project domains": project_domains[:5] if isinstance(project_domains, list) else [],  # Top 5 project names
        "Academic Projects": resume_obj.academic_projects or [],  # Academic Projects from model
        "List of Work context": work_context if isinstance(work_context, list) else [],  # Work context from model
        "Education": resume_obj.education or [],
        "Work Experience": experience_list if isinstance(experience_list, list) else [],
        "Professional Projects": projects_list if isinstance(projects_list, list) else [],
        "Certifications": resume_obj.certifications or [],
        "Total Experience": resume_obj.total_experience or 0,
    }
    
    return qse_resume


class InterviewSession:
    """
    Interview session using QSE (Question Selection Engine) for question generation
    and orchestration. All resume parsing is done via resume_parser.py.
    """
    def __init__(self, resume_obj, job_description_text, company=None, role=None, industry=None):
        self.resume_obj = resume_obj  # Save the Resume object
        self.job_description_text = job_description_text or "N/A"
        self.company = company or "N/A"
        self.role = role or "Data Scientist"
        self.industry = industry or "Technology"

        # Convert resume to QSE format using resume_parser if available
        self.qse_parsed_resume = convert_resume_to_qse_format(resume_obj)
        self.candidate_name = self.qse_parsed_resume.get("Name") or resume_obj.name or "Candidate"
        
        # Build question pool using QSE (this happens during initialization)
        if QSE_AVAILABLE:
            print(f"[InterviewSession] Building question pool for {self.candidate_name}...")
            print(f"[InterviewSession] Resume keys: {list(self.qse_parsed_resume.keys())}")
            print(f"[InterviewSession] Company: {self.company}, JD length: {len(self.job_description_text)}")
            try:
                # Get Pinecone index name from environment or use default
                pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "ds")
                print(f"[InterviewSession] Using Pinecone index: {pinecone_index_name}")
                
                self.scored_question_pool = build_scored_question_pool(
                    parsed_resume=self.qse_parsed_resume,
                    company_name=self.company,
                    job_description=self.job_description_text,
                    max_questions=20,
                    pinecone_index_name=pinecone_index_name,
                    n_perplexity_questions=10,
                )
                print(f"[InterviewSession] ✅ Generated {len(self.scored_question_pool)} scored questions")
                
                # Count questions by source for debugging
                source_counts = {}
                for q in self.scored_question_pool:
                    source = q.get("source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                print(f"[InterviewSession] Question sources: {source_counts}")
                if self.scored_question_pool:
                    print(f"[InterviewSession] Sample question: {self.scored_question_pool[0].get('question', 'N/A')[:100]}")
                else:
                    print("[InterviewSession] ⚠️ Warning: Question pool is empty!")
            except Exception as e:
                print(f"[InterviewSession] ❌ Error building question pool: {e}")
                import traceback
                traceback.print_exc()
                self.scored_question_pool = []
        else:
            print("[InterviewSession] ⚠️ QSE not available, using empty question pool")
            self.scored_question_pool = []

        # Orchestrator state
        self.interview_data = []  # List of {"question": str, "answer": str, "evaluation": str}
        self.smalltalk_history = []
        self.smalltalk_turn_count = 0
        self.smalltalk_completed = False
        self.off_topic_count = 0  # Off-topic count for small talk
        self.smalltalk_user_replies = 0  # Track number of user replies in small talk
        self.initial_greeting_sent = False  # Track if the initial greeting has been sent
        
        # Question state
        self.asked_questions = set()  # Track asked question texts
        self.current_question = None
        self.current_question_idx = 0
        self.follow_up_count = 0
        self.max_followups_per_question = 3
        self.total_main_questions_asked = 0
        self.max_main_questions = 1
        
        # Off-topic tracking for interview phase (separate from small talk)
        self.off_topic_strikes = 0  # Track off-topic/injection strikes during interview
        
        # Final evaluation
        self.full_evaluation = None

    def start_interview(self):
        """Start the interview with small talk"""
        if self.initial_greeting_sent:
            print(f"[start_interview] ⚠️ WARNING: start_interview called but greeting already sent!")
            return "Let's continue with the interview."
        
        print(f"[start_interview] QSE_AVAILABLE: {QSE_AVAILABLE}, Question pool size: {len(self.scored_question_pool) if hasattr(self, 'scored_question_pool') else 0}")
        
        if not QSE_AVAILABLE:
            # Fallback to old method
            greeting = f"Hi {self.candidate_name}, nice to meet you! Let's get started with your interview for the {self.role} role at {self.company}"
            print(f"[start_interview] Using fallback greeting: {greeting}")
            self.initial_greeting_sent = True
            return greeting
        
        # Start small talk using orchestrator
        print(f"[start_interview] Starting small talk for {self.candidate_name}, role: {self.role}, company: {self.company}")
        try:
            opener, history, turn_count = start_smalltalk(self.candidate_name, self.role, self.company)
            self.smalltalk_history = history
            self.smalltalk_turn_count = turn_count
            self.initial_greeting_sent = True
            print(f"[start_interview] Small talk started. Opener: {opener[:100]}...")
            return opener
        except Exception as e:
            print(f"[start_interview] Error in start_smalltalk: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            self.initial_greeting_sent = True
            return f"Hi {self.candidate_name}, nice to meet you! Let's get started with your interview for the {self.role} role at {self.company}"

    def process_smalltalk_reply(self, user_reply: str):
        """Process user reply during small talk phase - allows up to MAX_SMALLTALK_TURNS (3) conversational turns"""
        self.smalltalk_user_replies += 1
        print(f"[process_smalltalk_reply] Processing reply #{self.smalltalk_user_replies}, QSE_AVAILABLE: {QSE_AVAILABLE}, smalltalk_completed: {self.smalltalk_completed}, turn_count: {self.smalltalk_turn_count}")
        print(f"[process_smalltalk_reply] History length: {len(self.smalltalk_history)}, History: {self.smalltalk_history}")
        
        if not QSE_AVAILABLE:
            print("[process_smalltalk_reply] QSE not available, transitioning directly")
            self.smalltalk_completed = True
            first_q = self._get_first_interview_question()
            return f"Thanks! Let's get started with the interview.\n\n{first_q}", False, False
        
        # Use continue_smalltalk to handle full conversational flow (up to MAX_SMALLTALK_TURNS)
        try:
            print(f"[process_smalltalk_reply] Calling continue_smalltalk with history length: {len(self.smalltalk_history)}, turn_count: {self.smalltalk_turn_count}")
            
            # Check if history is empty or malformed
            if not self.smalltalk_history or len(self.smalltalk_history) == 0:
                print(f"[process_smalltalk_reply] ⚠️ WARNING: History is empty! Transitioning directly.")
                self.smalltalk_completed = True
                first_q = self._get_first_interview_question()
                return f"Thanks! Let's get started with the interview.\n\n{first_q}", False, False
            
            ai_msg, updated_history, turn_count, stop_smalltalk, terminate_all, off_topic_count = (
                continue_smalltalk(self.smalltalk_history, user_reply, self.smalltalk_turn_count, self.off_topic_count)
            )
            
            self.smalltalk_history = updated_history
            self.smalltalk_turn_count = turn_count
            self.off_topic_count = off_topic_count
            
            print(f"[process_smalltalk_reply] Result - stop_smalltalk: {stop_smalltalk}, terminate_all: {terminate_all}, turn_count: {turn_count}")
            print(f"[process_smalltalk_reply] AI message preview: {ai_msg[:100]}...")
            
            if terminate_all:
                return "It seems we're not staying on track, so I'll end the interview here. Thank you for your time.", True, True
            
            if stop_smalltalk:
                self.smalltalk_completed = True
                print("[process_smalltalk_reply] Small talk completed, getting first interview question...")
                first_q = self._get_first_interview_question()
                response = f"{ai_msg}\n\n{first_q}"
                print(f"[process_smalltalk_reply] ✅ Transition response: {response[:150]}...")
                return response, False, False
            
            # Continue small talk - return AI's conversational response
            return ai_msg, False, False
        except Exception as e:
            print(f"[process_smalltalk_reply] ❌ Error in continue_smalltalk: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: transition to interview
            self.smalltalk_completed = True
            first_q = self._get_first_interview_question()
            return f"Thanks! Let's get started with the interview.\n\n{first_q}", False, False

    def _get_first_interview_question(self) -> str:
        """Get the first interview question (resume-based)"""
        print(f"[_get_first_interview_question] QSE_AVAILABLE: {QSE_AVAILABLE}, Question pool size: {len(self.scored_question_pool)}")
        
        if not QSE_AVAILABLE:
            fallback_q = "Tell me about yourself and your experience."
            print(f"[_get_first_interview_question] Using fallback question: {fallback_q}")
            self.current_question = fallback_q
            self.asked_questions.add(fallback_q)
            self.interview_data.append({
                "question": fallback_q,
                "answer": None,
                "evaluation": None
            })
            self.total_main_questions_asked += 1
            return fallback_q
        
        # Check if we have questions in the pool
        if not self.scored_question_pool:
            print("[_get_first_interview_question] ⚠️ Warning: No questions in pool, using fallback")
            fallback_q = "Tell me about a recent project you worked on."
            self.current_question = fallback_q
            self.asked_questions.add(fallback_q)
            self.interview_data.append({
                "question": fallback_q,
                "answer": None,
                "evaluation": None
            })
            self.total_main_questions_asked += 1
            return fallback_q
        
        # Use orchestrator function
        try:
            rag_questions = [q.get("question", "") if isinstance(q, dict) else str(q) for q in self.scored_question_pool[:5]]
            print(f"[_get_first_interview_question] Using {len(rag_questions)} questions from pool for first question generation")
            first_q = get_first_interview_question(self.qse_parsed_resume, rag_questions)
            self.current_question = normalize_question(first_q)
            self.asked_questions.add(self.current_question)
            
            # Add to interview data
            self.interview_data.append({
                "question": self.current_question,
                "answer": None,
                "evaluation": None
            })
            self.total_main_questions_asked += 1
            
            print(f"[_get_first_interview_question] ✅ First question: {self.current_question[:100]}...")
            return self.current_question
        except Exception as e:
            print(f"[_get_first_interview_question] ❌ Error getting first question: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            fallback_q = "Tell me about yourself and your experience."
            self.current_question = fallback_q
            self.asked_questions.add(fallback_q)
            self.interview_data.append({
                "question": fallback_q,
                "answer": None,
                "evaluation": None
            })
            self.total_main_questions_asked += 1
            return fallback_q

    def _get_next_question_from_pool(self) -> Optional[str]:
        """Helper to get next question from pool without updating state (for skipped questions)"""
        if not QSE_AVAILABLE or not self.scored_question_pool:
            return None
        
        try:
            previous_q = self.current_question if self.current_question else ""
            next_q = get_next_diverse_question(
                question_pool=self.scored_question_pool,
                asked_set=self.asked_questions,
                previous_question=previous_q,
                similarity_threshold=0.80
            )
            
            if next_q:
                # Update state
                self.current_question = next_q
                self.asked_questions.add(next_q)
                self.interview_data.append({
                    "question": next_q,
                    "answer": None,
                    "evaluation": None
                })
                self.total_main_questions_asked += 1
                self.follow_up_count = 0
                return next_q
            return None
        except Exception as e:
            print(f"[_get_next_question_from_pool] Error: {e}")
            return None

    def evaluate_answer(self, answer, question=None):
        """Evaluate a single answer using the evaluation engine"""
        # Use the evaluation engine for a single question
        interview_payload = {
            "interview_id": f"session_{id(self)}",
            "candidate_name": self.candidate_name,
            "questions": [{
                "id": self.current_question_idx + 1,
                "question": question if question else self.current_question,
                "answer": answer,
                "is_icebreaker": False,
            }],
            "icebreaker_count": 0,
        }
        
        evaluation_result = evaluate_interview_json(interview_payload, use_llm=True)
        
        # Format feedback using evaluation engine helper
        return format_answer_evaluation_feedback(evaluation_result)

    def add_answer(self, answer):
        """Process user answer and determine next action"""
        # If still in small talk phase (shouldn't happen if views.py handles it correctly)
        if not self.smalltalk_completed:
            print(f"[add_answer] ⚠️ WARNING: add_answer called while smalltalk not completed! This should be handled by views.py")
            print(f"[add_answer] Processing smalltalk reply from add_answer...")
            ai_response, terminate, _ = self.process_smalltalk_reply(answer)
            if terminate:
                return ai_response, "Interview terminated."
            if not self.smalltalk_completed:
                return ai_response, "Small talk continued."
            # If smalltalk completed here, fall through to interview phase
            print(f"[add_answer] Smalltalk completed after processing, continuing to interview phase")
        
        # Now in interview phase - handle answer to current question
        if not self.current_question:
            # Shouldn't happen, but handle gracefully
            self._get_first_interview_question()
        
        # Classify answer behavior using orchestrator
        if QSE_AVAILABLE:
            behavior = classify_answer_behavior(self.current_question, answer)
            print(f"[add_answer] Answer behavior classified as: {behavior}")
            
            # Handle clarification requests
            if behavior == "clarification_request":
                clarified_q = clarify_question(self.current_question)
                self.current_question = clarified_q
                self.interview_data[self.current_question_idx]["question"] = clarified_q
                return clarified_q, "Question clarified."
            
            # Handle off-topic/injection responses with warning system
            if behavior == "off_topic_or_injection":
                self.off_topic_strikes += 1
                print(f"[add_answer] Off-topic/injection detected. Strike count: {self.off_topic_strikes}")
                
                if self.off_topic_strikes == 1:
                    # First strike: give warning and re-ask the same question
                    warning_msg = (
                        "I'd appreciate if we keep the discussion focused on the interview questions. "
                        "Let's try that again.\n\n"
                        f"{self.current_question}"
                    )
                    print(f"[add_answer] First strike - giving warning and re-asking question: {warning_msg}")
                    # Don't save answer, don't increment question idx - re-ask same question
                    return warning_msg, "Off-topic warning, re-asking question."
                else:
                    # Second strike: end interview
                    ending_msg = (
                        "It seems we're not staying on track, so I'll end the interview here. "
                        "Thank you for your time."
                    )
                    print(f"[add_answer] Second strike - ending interview: {ending_msg}")
                    # Mark question as ended
                    self.interview_data[self.current_question_idx]["answer"] = answer
                    self.interview_data[self.current_question_idx]["evaluation"] = "Interview ended due to off-topic/injection behavior"
                    return ending_msg, "Interview ended."
            
            # Handle "don't know" responses
            if behavior == "dont_know":
                # Get transition phrase and next question
                if QSE_AVAILABLE:
                    transition = transition_phrase("next_topic")
                else:
                    transition = "That's alright! Let's move on to something else."
                
                # Mark this question as skipped
                self.interview_data[self.current_question_idx]["answer"] = answer
                self.interview_data[self.current_question_idx]["evaluation"] = "Skipped - candidate indicated they didn't know"
                self.current_question_idx += 1
                self.follow_up_count = 0
                
                # Get next question
                next_q = self._get_next_question_from_pool()
                if next_q:
                    combined_response = f"{transition}\n\n{next_q}"
                    return combined_response, "Question skipped, next question provided."
                else:
                    return transition, "Question skipped."
        
        # Save the answer to the current question
        self.interview_data[self.current_question_idx]["answer"] = answer
        print(f"[add_answer] Saved answer to question idx {self.current_question_idx}: {self.current_question[:100] if self.current_question else 'None'}...")
        
        # Check if we need a follow-up
        needs_follow = False
        if QSE_AVAILABLE and self.follow_up_count < self.max_followups_per_question:
            needs_follow = needs_followup(answer, self.current_question)
            print(f"[add_answer] Follow-up needed: {needs_follow}, follow_up_count: {self.follow_up_count}")
        
        if needs_follow:
            # Generate follow-up question
            print(f"[add_answer] Generating follow-up question...")
            followup_q = generate_followup_question(answer, self.current_question)
            self.follow_up_count += 1
            
            # Evaluate the current answer first (proper evaluation)
            evaluation_feedback = self.evaluate_answer(answer, self.current_question)
            self.interview_data[self.current_question_idx]["evaluation"] = evaluation_feedback
            
            # Add follow-up question as a new entry
            self.current_question_idx += 1  # Move to next slot for follow-up
            self.current_question = followup_q
            self.interview_data.append({
                "question": followup_q,
                "answer": None,
                "evaluation": None
            })
            
            print(f"[add_answer] ✅ Follow-up question added at idx {self.current_question_idx}: {followup_q[:100]}...")
            return followup_q, "Follow-up question asked."
        else:
            # Evaluate the answer
            evaluation_feedback = self.evaluate_answer(answer, self.current_question)
            self.interview_data[self.current_question_idx]["evaluation"] = evaluation_feedback
            self.current_question_idx += 1  # Move to next question slot
            self.follow_up_count = 0  # Reset follow-up count for next question
            
            print(f"[add_answer] Answer evaluated, moving to next question. New idx: {self.current_question_idx}")
            return evaluation_feedback, "Answer recorded and evaluated."

    def next_question(self, interview_type="general"):
        """Get the next question from the pool"""
        # Check if we've reached max questions
        if self.total_main_questions_asked >= self.max_main_questions:
            return "We've reached the end of your interview. Thank you for your time!"

        # If no current question, get first one
        if not self.current_question:
            if not self.smalltalk_completed:
                return "Please complete the initial conversation first."
            self._get_first_interview_question()
            return self.current_question
        
        # Get next diverse question from pool
        if QSE_AVAILABLE and self.scored_question_pool:
            previous_q = self.current_question
            next_q = get_next_diverse_question(
                question_pool=self.scored_question_pool,
                asked_set=self.asked_questions,
                previous_question=previous_q,
                similarity_threshold=0.80
            )
            
            if not next_q:
                return "We've reached the end of your interview. Thank you for your time!"
            
            self.current_question = next_q
            self.asked_questions.add(next_q)
            self.interview_data.append({
                "question": next_q,
                "answer": None,
                "evaluation": None
            })
            self.total_main_questions_asked += 1
            self.follow_up_count = 0

            return next_q
        else:
            # Fallback
            return "We've completed all available questions. Thank you for your time!"

    def get_interview_summary(self):
        """Generate comprehensive interview summary using the evaluation engine"""
        # Prepare questions for evaluation (exclude small talk, only technical questions)
        questions_for_eval = []
        question_id = 1
        for q_data in self.interview_data:
            if q_data.get("answer") and q_data.get("answer") != "":  # Only include answered questions
                questions_for_eval.append({
                    "id": question_id,
                    "question": q_data["question"],
                    "answer": q_data["answer"],
                    "is_icebreaker": False,  # Small talk is separate, all interview questions are technical
                })
                question_id += 1
        
        if not questions_for_eval:
            return "No interview questions were answered."
        
        # Get comprehensive evaluation
        interview_payload = {
            "interview_id": f"session_{id(self)}",
            "candidate_name": self.candidate_name,
            "questions": questions_for_eval,
            "icebreaker_count": 0,  # Small talk is not included in evaluation
        }
        
        evaluation_result = evaluate_interview_json(interview_payload, use_llm=True)
        
        # Store the full evaluation result for later use
        self.full_evaluation = evaluation_result
        
        # Format summary using evaluation engine helper
        return format_interview_summary(evaluation_result)
    
    @property
    def total_questions_asked(self):
        """Compatibility property for views.py"""
        return self.total_main_questions_asked
