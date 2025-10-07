from django import forms
from .models import Resume, Interview
from .script import extract_resume_structure

class ResumeUploadForm(forms.ModelForm):
	class Meta:
		model = Resume
		fields = ['file']

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['file'].required = False

	def save(self, commit=True, user=None):
		resume = super().save(commit=False)
		if user:
			resume.user = user

		if commit:
			resume.save()
			resume.refresh_from_db()

			structured = extract_resume_structure(resume.file.path)

			resume.name = structured['contact_info'].get('name')
			resume.email = structured['contact_info'].get('email')
			resume.phone = structured['contact_info'].get('phone')
			resume.location = structured['contact_info'].get('location')
			resume.summary = structured.get('summary')
			resume.education = structured.get('education')
			resume.experience = structured.get('experience')
			resume.skills = structured.get('skills')
			resume.certifications = structured.get('certifications')
			resume.projects = structured.get('projects')
			resume.domain_expertise = structured.get('domain_expertise')
			resume.years_of_experience = structured.get('years_of_experience', 0)

			resume.save()  # save enriched fields
		return resume



class ResumeFormForInterview(ResumeUploadForm):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['file'].widget.attrs.update({'class': 'w-full'})



# class InterviewForm(forms.ModelForm):
#     class Meta:
#         model = Interview
#         fields = ['company', 'job_description_url', 'resume', 'interview_type', 'duration']
#         widgets = {
#             'company': forms.TextInput(attrs={'placeholder': 'Company', 'class': 'w-full'}),
#             'job_description_url': forms.URLInput(attrs={'placeholder': 'Job Description URL', 'class': 'w-full'}),
#             'interview_type': forms.Select(attrs={'class': 'w-full'}),
#             'duration': forms.Select(attrs={'class': 'w-full'}),
#             'resume': forms.Select(attrs={'class': 'w-full'}),
#         }

class InterviewForm(forms.ModelForm):
	class Meta:
		model = Interview
		fields = ['company', 'job_description', 'resume', 'interview_type', 'duration']
		widgets = {
			'company': forms.TextInput(attrs={
				'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm focus:ring-orange-500 focus:ring-2',
				'placeholder': 'e.g. Google'
			}),
			'job_description': forms.Textarea(attrs={
				'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm',
				'rows': 4,
				'placeholder': 'Paste key bullets or a link to the JD'
			}),
			'resume': forms.Select(attrs={
				'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm'
			}),
			'interview_type': forms.Select(attrs={
				'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm'
			}),
			'duration': forms.Select(attrs={
				'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm'
			}),
		}

