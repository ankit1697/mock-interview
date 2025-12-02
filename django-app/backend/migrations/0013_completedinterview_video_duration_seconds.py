# Generated migration to add video_duration_seconds field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0012_completedinterview_visual_feedback'),
    ]

    operations = [
        migrations.AddField(
            model_name='completedinterview',
            name='video_duration_seconds',
            field=models.PositiveIntegerField(blank=True, help_text='Duration of the video recording in seconds', null=True),
        ),
    ]
