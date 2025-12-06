from django import forms
from .models import Resume, Interview
import sys
from pathlib import Path

# Add cgpt directory to path for resume_parser import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CGPT_PATH = PROJECT_ROOT / 'cgpt'
if str(CGPT_PATH) not in sys.path:
    sys.path.insert(0, str(CGPT_PATH))

try:
    from resume_parser import parse_resume_from_file
    RESUME_PARSER_AVAILABLE = True
except ImportError:
    RESUME_PARSER_AVAILABLE = False
    print("Warning: resume_parser not available in forms.py")

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

			# Use resume_parser if available
			if RESUME_PARSER_AVAILABLE and resume.file:
				try:
					structured = parse_resume_from_file(resume.file.path)
					
					# Map resume_parser.py output format to Django model fields
					# resume_parser returns: Name, Professional Summary, Professional Projects, Skills, 
					# Academic Projects, Work Experience, Total Experience, List of Core skills, 
					# List of Project domains, List of Work context
					
					resume.name = structured.get('Name') or structured.get('name') or None
					resume.summary = structured.get('Professional Summary') or structured.get('summary') or None
					resume.professional_projects = structured.get('Professional Projects') or structured.get('professional_projects') or []
					
					# Handle skills - resume_parser may return a list or dict
					skills_data = structured.get('Skills') or structured.get('skills') or {}
					if isinstance(skills_data, dict):
						resume.skills = skills_data
					elif isinstance(skills_data, list):
						resume.skills = {"technical": skills_data}
					else:
						resume.skills = {}
					
					resume.academic_projects = structured.get('Academic Projects') or structured.get('academic_projects') or []
					resume.work_experience = structured.get('Work Experience') or structured.get('work_experience') or []
					resume.total_experience = structured.get('Total Experience') or structured.get('total_experience') or 0
					resume.core_skills = structured.get('List of Core skills') or structured.get('core_skills') or []
					resume.project_domains = structured.get('List of Project domains') or structured.get('project_domains') or []
					resume.work_context = structured.get('List of Work context') or structured.get('work_context') or []
					
					# Legacy/optional fields (if present in parser output)
					resume.education = structured.get('Education') or structured.get('education') or resume.education
					resume.certifications = structured.get('Certifications') or structured.get('certifications') or resume.certifications
					resume.domain_expertise = structured.get('domain_expertise') or resume.domain_expertise
					
				except Exception as e:
					print(f"Error parsing resume with resume_parser: {e}")
					import traceback
					traceback.print_exc()
					# Leave fields as None if parsing fails

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
		fields = ['company', 'job_description', 'resume']
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
		}

