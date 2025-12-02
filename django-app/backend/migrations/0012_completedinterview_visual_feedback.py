# Generated migration to add visual_feedback field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0011_completedinterview_video_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='completedinterview',
            name='visual_feedback',
            field=models.TextField(blank=True, help_text='Visual behavior analysis feedback from video', null=True),
        ),
    ]
