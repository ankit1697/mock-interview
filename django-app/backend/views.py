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
from .utils import read_file
from PyPDF2 import PdfReader
from .script import InterviewSession


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
		upload_form = ResumeUploadForm()

	return render(request, 'interview_setup.html', {
		'form': interview_form,
		'upload_form': upload_form
	})




active_sessions = {}

def extract_text_from_pdf(path):
	reader = PdfReader(path)
	return "\n".join([page.extract_text() or "" for page in reader.pages])

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

	# (rest of your POST handling and render remains same!)


	if request.method == "POST":
	    user_message = request.POST.get("message", "").strip()

	    if not user_message:
	        return JsonResponse({"response": "Please enter a valid message."})

	    if session.total_questions_asked == 0 and session.current_question is None:
	        greeting_and_first_q = session.start_interview()
	        return JsonResponse({"response": greeting_and_first_q})

	    # First: Add user's answer
	    feedback_or_next_question, note = session.add_answer(user_message)

	    if note == "Question clarified.":
	        return JsonResponse({"response": feedback_or_next_question})

	    # Now carefully handle moving to the next step
	    next_question = session.next_question()

	    # 🛠 NEW LOGIC: if interview ended
	    if "end of your interview" in next_question.lower():
	        if session.interview_data[-1]["answer"] is None:
	            session.interview_data[-1]["answer"] = user_message
	            session.interview_data[-1]["evaluation"] = feedback_or_next_question

	        overall_summary = session.get_interview_summary()

	        completed_interview = CompletedInterview.objects.create(
	            interview=interview,
	            user=request.user,
	            transcript=json.dumps(session.interview_data, indent=2),
	            summary=overall_summary
	        )
	        active_sessions.pop(session_key, None)

	        return JsonResponse({
	            "response": next_question,
	            "completed_id": completed_interview.id
	        })

	    # 🛠 Important: otherwise return NEXT QUESTION cleanly
	    return JsonResponse({
	        "response": next_question
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