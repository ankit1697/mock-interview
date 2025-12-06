from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField

class Resume(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes')
	file = models.FileField(upload_to='resumes/')
	uploaded_at = models.DateTimeField(auto_now_add=True)
	is_preferred = models.BooleanField(default=False)

	# Parsed Fields (matching resume_parser.py output)
	name = models.CharField(max_length=255, null=True, blank=True, help_text="Candidate name from resume")
	summary = models.TextField(null=True, blank=True, help_text="Professional Summary")
	professional_projects = models.JSONField(null=True, blank=True, help_text="Professional Projects")
	skills = models.JSONField(null=True, blank=True, help_text="Skills")
	academic_projects = models.JSONField(null=True, blank=True, help_text="Academic Projects")
	work_experience = models.JSONField(null=True, blank=True, help_text="Work Experience")
	total_experience = models.FloatField(default=0, help_text="Total Experience (in years)")
	core_skills = models.JSONField(null=True, blank=True, help_text="List of Core skills (e.g., 'XGBoost', 'Forecasting', 'Optimization')")
	project_domains = models.JSONField(null=True, blank=True, help_text="List of Project domains (e.g., 'Donor forecasting model', 'Route optimization')")
	work_context = models.JSONField(null=True, blank=True, help_text="List of Work context (e.g., 'BioLife Plasma Services', 'Healthcare analytics')")
	
	# Legacy fields (kept for backward compatibility, may be populated from other sources)
	education = models.JSONField(null=True, blank=True)
	certifications = models.JSONField(null=True, blank=True)
	domain_expertise = models.JSONField(null=True, blank=True)

	def __str__(self):
		return f"{self.user.username} - {self.file.name}"

	class Meta:
		ordering = ['-is_preferred', '-uploaded_at']


class Interview(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="interviews")
	company = models.CharField(max_length=255)
	job_description = models.CharField(max_length=2000,blank=True)
	resume = models.ForeignKey(Resume, on_delete=models.SET_NULL, null=True, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"{self.user.username} - {self.company}"


class CompletedInterview(models.Model):
	interview = models.OneToOneField('Interview', on_delete=models.CASCADE)
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	transcript = models.TextField()
	summary = models.TextField(null=True, blank=True)
	evaluation_results = models.JSONField(null=True, blank=True)  # Full structured evaluation JSON
	# Aggregate numeric scores
	overall_score = models.FloatField(null=True, blank=True)
	technical_reasoning_avg = models.FloatField(null=True, blank=True)
	accuracy_avg = models.FloatField(null=True, blank=True)
	confidence_avg = models.FloatField(null=True, blank=True)
	problem_solving_avg = models.FloatField(null=True, blank=True)
	flow_avg = models.FloatField(null=True, blank=True)
	video_file = models.FileField(upload_to='interview_videos/', null=True, blank=True, help_text='Compressed video recording of the interview')
	video_duration_seconds = models.PositiveIntegerField(null=True, blank=True, help_text='Duration of the video recording in seconds')
	visual_feedback = models.TextField(null=True, blank=True, help_text='Visual behavior analysis feedback from video')
	# Model prediction scores (0-1 scale)
	engagement_score = models.FloatField(null=True, blank=True, help_text='Engagement score from vision model')
	engaging_tone_score = models.FloatField(null=True, blank=True, help_text='Engaging Tone score from vision model')
	excitement_score = models.FloatField(null=True, blank=True, help_text='Excitement score from vision model')
	friendliness_score = models.FloatField(null=True, blank=True, help_text='Friendliness score from vision model')
	smile_score = models.FloatField(null=True, blank=True, help_text='Smile score from vision model')
	completed_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"Interview #{self.interview.id} by {self.user.username}"
