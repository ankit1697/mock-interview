from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Resume, Interview, CompletedInterview
from .forms import ResumeUploadForm, InterviewForm
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
import os
import json
from difflib import SequenceMatcher
from .utils import read_file, transcribe_audio, text_to_speech
from PyPDF2 import PdfReader
from .script import InterviewSession
import requests
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required


# Minimal endpoint to create an ephemeral realtime session/token with OpenAI
@login_required
@require_POST
@csrf_exempt
def create_realtime_session(request):
    """Issue an ephemeral realtime session from OpenAI and return the provider response.

    This keeps the long-lived OpenAI API key on the server. The client should call
    this endpoint to get short-lived credentials and then use those credentials to
    establish a direct realtime (WebRTC) connection to the provider.
    """
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_KEY:
        return JsonResponse({"error": "Server missing OpenAI API key"}, status=500)

    try:
        # Adjust this URL/payload to match the realtime provider's ephemeral session API
        url = "https://api.openai.com/v1/realtime/sessions"
        payload = {"model": "gpt-4o-realtime-preview-2024", "voice": "alloy"}
        headers = {
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return JsonResponse(resp.json())
    except requests.RequestException as e:
        return JsonResponse({"error": f"Could not create realtime session: {str(e)}"}, status=502)


def login_view(request):
	return render(request, 'login.html')

def index(request):
	return render(request, 'index.html')

@login_required
def profile(request):
	resumes = Resume.objects.filter(user=request.user)
	form = ResumeUploadForm(request.POST or None, request.FILES or None)

	if request.method == 'POST' and form.is_valid():
		resume = form.save(user=request.user)
		print("Resume saved:", resume.file.name)
		return redirect('profile')

	return render(request, 'profile.html', {
		'form': form,
		'resumes': resumes
	})


@require_POST
@login_required
def set_preferred_resume(request, resume_id):
	resume = get_object_or_404(Resume, id=resume_id, user=request.user)

	# Unset existing preferred resumes
	Resume.objects.filter(user=request.user, is_preferred=True).update(is_preferred=False)

	resume.is_preferred = True
	resume.save()

	return redirect('profile')


@require_POST
@login_required
def delete_resume(request, resume_id):
	resume = get_object_or_404(Resume, id=resume_id, user=request.user)
	resume.file.delete()  # delete the file itself
	resume.delete()
	return redirect('profile')


@require_POST
@login_required
def upload_resume(request):
	if request.method == 'POST':
		form = ResumeUploadForm(request.POST, request.FILES)
		if form.is_valid():
			form.save(user=request.user)
			return redirect('profile')
	else:
		form = ResumeUploadForm()
	return render(request, 'profile.html', {'form': form})



@login_required
def setup_interview(request):
	preferred_resume = request.user.resumes.filter(is_preferred=True).first()

	if request.method == "POST":
		interview_form = InterviewForm(request.POST)
		upload_form = ResumeUploadForm(request.POST, request.FILES)

		selected_resume = None

		# handle resume upload first
		if upload_form.is_valid() and 'file' in request.FILES:
			uploaded_resume = upload_form.save(commit=False)
			uploaded_resume.user = request.user
			uploaded_resume.save()
			selected_resume = uploaded_resume

			# Rebind interview form so the new resume is selected in the dropdown
			data = interview_form.data.copy()
			data['resume'] = str(uploaded_resume.id)
			interview_form = InterviewForm(data)

		# Ensure resume field is limited to user's resumes and remove blank option
		interview_form.fields['resume'].queryset = request.user.resumes.all()
		interview_form.fields['resume'].empty_label = None

		if interview_form.is_valid():
			interview = interview_form.save(commit=False)
			interview.user = request.user
			if selected_resume:
				interview.resume = selected_resume
			interview.save()
			return redirect('interview_chat', interview_id=interview.id)

	else:
		interview_form = InterviewForm(initial={
			'resume': preferred_resume.id if preferred_resume else None
		})
		interview_form.fields['resume'].queryset = request.user.resumes.all()
		interview_form.fields['resume'].empty_label = None
		upload_form = ResumeUploadForm()

	return render(request, 'interview_setup.html', {
		'form': interview_form,
		'upload_form': upload_form
	})




active_sessions = {}

def extract_text_from_pdf(path):
	reader = PdfReader(path)
	return "\n".join([page.extract_text() or "" for page in reader.pages])


def _compute_dimension_averages(evaluation_result):
    """
    Compute overall and per-dimension average scores from the structured
    evaluation JSON produced by evaluation_engine.evaluate_interview_json.
    """
    if not evaluation_result:
        return {
            "overall_score": None,
            "technical_reasoning_avg": None,
            "accuracy_avg": None,
            "confidence_avg": None,
            "problem_solving_avg": None,
            "flow_avg": None,
        }

    results = [
        r for r in (evaluation_result.get("results") or [])
        if not r.get("skipped")
    ]
    if not results:
        return {
            "overall_score": evaluation_result.get("overall_score"),
            "technical_reasoning_avg": None,
            "accuracy_avg": None,
            "confidence_avg": None,
            "problem_solving_avg": None,
            "flow_avg": None,
        }

    dims = ["technical_reasoning", "accuracy", "confidence", "problem_solving", "flow"]
    sums = {d: 0.0 for d in dims}
    count = 0

    for r in results:
        subs = r.get("subscores") or {}
        for d in dims:
            val = subs.get(d)
            if val is not None:
                sums[d] += float(val)
        count += 1

    if count == 0:
        return {
            "overall_score": evaluation_result.get("overall_score"),
            "technical_reasoning_avg": None,
            "accuracy_avg": None,
            "confidence_avg": None,
            "problem_solving_avg": None,
            "flow_avg": None,
        }

    avgs = {d: round(sums[d] / count, 3) for d in dims}

    return {
        "overall_score": evaluation_result.get("overall_score"),
        "technical_reasoning_avg": avgs["technical_reasoning"],
        "accuracy_avg": avgs["accuracy"],
        "confidence_avg": avgs["confidence"],
        "problem_solving_avg": avgs["problem_solving"],
        "flow_avg": avgs["flow"],
    }

@login_required
@csrf_exempt
def interview_chat(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id, user=request.user)

    session_key = f"session_{request.user.id}_{interview_id}"
    session = active_sessions.get(session_key)

    if not session:
        resume = interview.resume
        job_description_text = interview.job_description or "N/A"
        company = interview.company or "N/A"
        session = InterviewSession(resume, job_description_text, company)
        active_sessions[session_key] = session

    if request.method == "POST":
        # Handle audio input
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            user_message = transcribe_audio(audio_file)
            if not user_message:
                return JsonResponse({"error": "Could not transcribe audio"}, status=400)
        else:
            user_message = request.POST.get("message", "").strip()
        # If client requested an explicit stop, finalize the interview and return completed_id
        if request.POST.get('stop') or user_message == '__stop__':
            print(f"[interview_chat] Stop requested for interview_id={interview_id} by user={request.user.id}")
            # Save final answer if provided
            if user_message and user_message not in ['__stop__', '__end__']:
                try:
                    session.interview_data[session.current_question_idx]["answer"] = user_message
                except Exception:
                    pass

            # Safety: if the last question slot exists but has no answer, and we have any recent answer stored in the session state, keep it
            try:
                if session.interview_data:
                    last_slot = session.interview_data[-1]
                    if last_slot.get("answer") is None and session.current_question_idx < len(session.interview_data):
                        current_slot = session.interview_data[session.current_question_idx]
                        if current_slot.get("answer"):
                            last_slot["answer"] = current_slot.get("answer")
            except Exception:
                pass

            overall_summary = session.get_interview_summary()
            print(f"[interview_chat] Generated interview summary for interview_id={interview_id}; creating CompletedInterview...")

            score_data = _compute_dimension_averages(session.full_evaluation)

            completed_interview = CompletedInterview.objects.create(
                interview=interview,
                user=request.user,
                transcript=json.dumps(session.interview_data, indent=2),
                summary=overall_summary,
                evaluation_results=session.full_evaluation,  # Save the full evaluation JSON
                overall_score=score_data["overall_score"],
                technical_reasoning_avg=score_data["technical_reasoning_avg"],
                accuracy_avg=score_data["accuracy_avg"],
                confidence_avg=score_data["confidence_avg"],
                problem_solving_avg=score_data["problem_solving_avg"],
                flow_avg=score_data["flow_avg"],
            )
            active_sessions.pop(session_key, None)

            audio_response = text_to_speech("Your interview has been stopped. Redirecting to feedback.")

            return JsonResponse({
                "transcript": user_message,
                "response": "Interview stopped. Redirecting...",
                "audio": audio_response,
                "completed_id": completed_interview.id
            })

        if not user_message:
            return JsonResponse({"response": "Please enter a valid message."})

        if session.total_questions_asked == 0 and session.current_question is None:
            greeting_and_first_q = session.start_interview()
            audio_response = text_to_speech(greeting_and_first_q)
            return JsonResponse({
                "transcript": "",  # No user message for initial greeting
                "response": greeting_and_first_q,
                "audio": audio_response
            })

        # First: Add user's answer
        feedback_or_next_question, note = session.add_answer(user_message)

        if note == "Question clarified.":
            audio_response = text_to_speech(feedback_or_next_question)
            return JsonResponse({
                "transcript": user_message,  # Include user's transcript
                "response": feedback_or_next_question,
                "audio": audio_response
            })

        # Handle moving to next step
        next_question = session.next_question()
        audio_response = text_to_speech(next_question)

        # Handle interview end
        if "end of your interview" in next_question.lower():
            if session.interview_data[-1]["answer"] is None:
                session.interview_data[-1]["answer"] = user_message
                session.interview_data[-1]["evaluation"] = feedback_or_next_question

            overall_summary = session.get_interview_summary()
            score_data = _compute_dimension_averages(session.full_evaluation)

            completed_interview = CompletedInterview.objects.create(
                interview=interview,
                user=request.user,
                transcript=json.dumps(session.interview_data, indent=2),
                summary=overall_summary,
                evaluation_results=session.full_evaluation,  # Save the full evaluation JSON
                overall_score=score_data["overall_score"],
                technical_reasoning_avg=score_data["technical_reasoning_avg"],
                accuracy_avg=score_data["accuracy_avg"],
                confidence_avg=score_data["confidence_avg"],
                problem_solving_avg=score_data["problem_solving_avg"],
                flow_avg=score_data["flow_avg"],
            )
            active_sessions.pop(session_key, None)

            return JsonResponse({
                "transcript": user_message,
                "response": next_question,
                "audio": audio_response,
                "completed_id": completed_interview.id
            })

        # Return next question with audio
        return JsonResponse({
            "transcript": user_message,  # Always include the transcript
            "response": next_question,
            "audio": audio_response
        })

    return render(request, "interview_chat.html", {"interview": interview})


@login_required
def past_interviews(request):
    completed_interviews = CompletedInterview.objects.filter(user=request.user).select_related('interview').order_by('-completed_at')
    return render(request, "past_interviews.html", {"completed_interviews": completed_interviews})


@login_required
def interview_feedback(request, completed_id):
    completed = get_object_or_404(CompletedInterview, id=completed_id, user=request.user)

    # Parse the transcript (it was saved as JSON text)
    try:
        transcript = json.loads(completed.transcript)
    except Exception:
        transcript = []

    return render(request, "interview_feedback.html", {
        "completed": completed,
        "transcript": transcript
    })


##################### OLD WORKING LOGIC #######################

# openai.api_key = os.getenv("OPENAI_API_KEY")

# chat_history_store = {}
# side_questions_store = {}

# def extract_text_from_pdf(path):
# 	reader = PdfReader(path)
# 	return "\n".join([page.extract_text() or "" for page in reader.pages])

# @login_required
# @csrf_exempt
# def interview_chat(request, interview_id):
# 	interview = get_object_or_404(Interview, id=interview_id, user=request.user)
# 	resume_path = interview.resume.file.path
# 	resume_text = extract_text_from_pdf(resume_path)
# 	job_description = interview.job_description or "N/A"

# 	chat_key = f"chat_{request.user.id}_{interview_id}"
# 	chat_history = chat_history_store.get(chat_key, [])
# 	side_questions = side_questions_store.get(chat_key, [])

# 	if request.method == "POST":
# 		message = request.POST.get("message")

# 		# If it's the first user message
# 		if len(chat_history) == 0:
# 			# Inject system + initial prompt
# 			chat_history.append({"role": "system", "content": "You are an expert AI interviewer for data science roles. Start with a friendly greeting, and then begin the interview."})
# 			chat_history.append({"role": "user", "content": f"Begin the interview based on the following resume and job description. Generate exactly 1 concise and thoughtful interview question. Do not include any explanations.\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}"})

# 			response = openai.ChatCompletion.create(
# 				model="gpt-4o",
# 				messages=chat_history
# 			)
# 			initial_question = response.choices[0].message.content.strip()
# 			chat_history.append({"role": "assistant", "content": initial_question})

# 			# Generate follow-up questions (store them separately)
# 			chat_history.append({"role": "user", "content": "Generate 3 follow-up questions based on the initial question. Return them as a numbered list with no additional explanation."})
# 			followup = openai.ChatCompletion.create(
# 				model="gpt-4o",
# 				messages=chat_history
# 			)
# 			raw_followups = followup.choices[0].message.content.strip().split('\n')
# 			side_questions = [
# 				line.strip().split('. ', 1)[-1] for line in raw_followups if line.strip() and '.' in line
# 			]

# 			chat_history_store[chat_key] = chat_history
# 			side_questions_store[chat_key] = side_questions

# 			return JsonResponse({"response": initial_question})

# 		# Continue conversation
# 		chat_history.append({"role": "user", "content": message})

# 		# Load follow-up queue
# 		side_questions = side_questions_store.get(chat_key, [])
# 		if side_questions:
# 			next_q = side_questions.pop(0)
# 			chat_history.append({"role": "assistant", "content": next_q})

# 			chat_history_store[chat_key] = chat_history
# 			side_questions_store[chat_key] = side_questions

# 			return JsonResponse({"response": next_q})
# 		else:
# 			closing_message = "Thanks for answering all the questions! The interview is now complete."
# 			chat_history.append({"role": "assistant", "content": closing_message})
# 			chat_history_store[chat_key] = chat_history
# 			CompletedInterview.objects.create(
# 				interview=interview,
# 				user=request.user,
# 				transcript=json.dumps(chat_history, indent=2)
# 			)
# 			return JsonResponse({"response": closing_message})

# 	return render(request, "interview_chat.html", {"interview": interview})