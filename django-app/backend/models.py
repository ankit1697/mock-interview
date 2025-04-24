from django.db import models
from django.contrib.auth.models import User

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='resumes')
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_preferred = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"

    class Meta:
        ordering = ['-is_preferred', '-uploaded_at']  # Preferred first


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
    transcript = models.TextField()  # JSON or plain text
    completed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Interview #{self.interview.id} by {self.user.username}"
