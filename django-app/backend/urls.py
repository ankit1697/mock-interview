from django.urls import path
from .views import *


urlpatterns = [
    path('', index, name='index'),
    path('profile/', profile, name='profile'),
    path('resume/set-preferred/<int:resume_id>/', set_preferred_resume, name='set_preferred_resume'),
    path('resume/delete/<int:resume_id>/', delete_resume, name='delete_resume'),
    path('interview/setup/', setup_interview, name='setup_interview'),
    path("interview/chat/<int:interview_id>/", interview_chat, name="interview_chat"),
    path('profile/past-interviews/', past_interviews, name='past_interviews'),
    path('past-interview/<int:completed_id>/', past_interviews, name='past_interview_detail'),

]
