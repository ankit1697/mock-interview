from django import forms
from .models import Resume, Interview


class ResumeUploadForm(forms.ModelForm):
    class Meta:
        model = Resume
        fields = ['file']


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
            'job_description_url': forms.TextInput(attrs={
                'class': 'w-full border border-gray-300 rounded px-3 py-2 text-sm'
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

