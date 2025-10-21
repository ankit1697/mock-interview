from django.urls import path
from django.shortcuts import redirect
from .views import *


def direct_google_login(request):
    """Redirect directly to Google OAuth login"""
    return redirect('/accounts/google/login/?process=login&next=/')

urlpatterns = [
    path('', index, name='index'),
    path('profile/', profile, name='profile'),
    path('resume/set-preferred/<int:resume_id>/', set_preferred_resume, name='set_preferred_resume'),
    path('resume/delete/<int:resume_id>/', delete_resume, name='delete_resume'),
    path('interview/setup/', setup_interview, name='setup_interview'),
    path("interview/chat/<int:interview_id>/", interview_chat, name="interview_chat"),
    path('profile/past-interviews/', past_interviews, name='past_interviews'),
    path('interview/feedback/<int:completed_id>/', interview_feedback, name='interview_feedback'),
    # Ephemeral realtime session (server issues short-lived credentials)
    path('realtime/session/', create_realtime_session, name='create_realtime_session'),
    # Direct Google login
    path('login/', direct_google_login, name='direct_google_login'),

]
