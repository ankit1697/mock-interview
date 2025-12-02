from django.contrib import admin
import json
from django.utils.safestring import mark_safe
from .models import CompletedInterview, Resume

@admin.register(CompletedInterview)
class CompletedInterviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'interview', 'completed_at', 'has_video')
    readonly_fields = ('interview', 'user', 'formatted_transcript', 'completed_at', 'video_preview')
    
    fieldsets = (
        ('Interview Details', {
            'fields': ('interview', 'user', 'completed_at')
        }),
        ('Content', {
            'fields': ('summary', 'transcript', 'formatted_transcript')
        }),
        ('Scores', {
            'fields': ('overall_score', 'technical_reasoning_avg', 'accuracy_avg', 'confidence_avg', 'problem_solving_avg', 'flow_avg')
        }),
        ('Media', {
            'fields': ('video_file', 'video_preview')
        }),
    )

    def formatted_transcript(self, obj):
        data = json.loads(obj.transcript)
        html = '<pre style="white-space: pre-wrap;">' + json.dumps(data, indent=2) + '</pre>'
        return mark_safe(html)

    formatted_transcript.short_description = 'Transcript'
    
    def has_video(self, obj):
        return bool(obj.video_file)
    has_video.boolean = True
    has_video.short_description = 'Has Video'
    
    def video_preview(self, obj):
        if obj.video_file:
            return mark_safe(f'<video controls width="400" src="{obj.video_file.url}"></video>')
        return "No video uploaded"
    video_preview.short_description = 'Video Preview'


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'file', 'is_preferred', 'uploaded_at')
    list_filter = ('is_preferred',)
    search_fields = ('user__username',)
