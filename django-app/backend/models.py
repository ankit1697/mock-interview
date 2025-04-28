from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField

class Resume(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes')
	file = models.FileField(upload_to='resumes/')
	uploaded_at = models.DateTimeField(auto_now_add=True)
	is_preferred = models.BooleanField(default=False)

	# Parsed Fields
	name = models.CharField(max_length=255, null=True, blank=True)
	email = models.EmailField(null=True, blank=True)
	phone = models.CharField(max_length=50, null=True, blank=True)
	location = models.CharField(max_length=255, null=True, blank=True)
	summary = models.TextField(null=True, blank=True)
	education = models.JSONField(null=True, blank=True)
	experience = models.JSONField(null=True, blank=True)
	skills = models.JSONField(null=True, blank=True)
	certifications = models.JSONField(null=True, blank=True)
	projects = models.JSONField(null=True, blank=True)
	domain_expertise = models.JSONField(null=True, blank=True)
	years_of_experience = models.FloatField(default=0)

	def __str__(self):
		return f"{self.user.username} - {self.file.name}"

	class Meta:
		ordering = ['-is_preferred', '-uploaded_at']



INTERVIEW_TYPES = [
	("technical", "Technical Case"),
	("behavioral", "Behavioral Round"),
	("system", "System Design"),
]

DURATION_CHOICES = [
	(15, "15 minutes"),
	(30, "30 minutes"),
	(45, "45 minutes"),
	(60, "60 minutes"),
]

class Interview(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="interviews")
	company = models.CharField(max_length=255)
	job_description = models.CharField(max_length=2000,blank=True)
	resume = models.ForeignKey(Resume, on_delete=models.SET_NULL, null=True, blank=True)
	interview_type = models.CharField(max_length=20, choices=INTERVIEW_TYPES)
	duration = models.PositiveIntegerField(choices=DURATION_CHOICES)
	created_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"{self.user.username} - {self.company} ({self.get_interview_type_display()})"


class CompletedInterview(models.Model):
	interview = models.OneToOneField('Interview', on_delete=models.CASCADE)
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	transcript = models.TextField()
	summary = models.TextField(null=True, blank=True)
	completed_at = models.DateTimeField(auto_now_add=True)

	def __str__(self):
		return f"Interview #{self.interview.id} by {self.user.username}"
