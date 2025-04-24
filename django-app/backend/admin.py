from django.contrib import admin
import json
from django.utils.safestring import mark_safe
from .models import CompletedInterview, Resume

@admin.register(CompletedInterview)
class CompletedInterviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'interview', 'completed_at')
    readonly_fields = ('interview', 'user', 'formatted_transcript', 'completed_at')

    def formatted_transcript(self, obj):
        data = json.loads(obj.transcript)
        html = '<pre style="white-space: pre-wrap;">' + json.dumps(data, indent=2) + '</pre>'
        return mark_safe(html)

    formatted_transcript.short_description = 'Transcript'


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'file', 'is_preferred', 'uploaded_at')
    list_filter = ('is_preferred',)
    search_fields = ('user__username',)
