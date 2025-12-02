# Generated migration to add video_file field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0010_completedinterview_scores'),
    ]

    operations = [
        migrations.AddField(
            model_name='completedinterview',
            name='video_file',
            field=models.FileField(blank=True, help_text='Compressed video recording of the interview', null=True, upload_to='interview_videos/'),
        ),
    ]
