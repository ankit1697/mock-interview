# Generated migration to remove interview_type and duration fields

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0008_completedinterview_evaluation_results'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='interview',
            name='interview_type',
        ),
        migrations.RemoveField(
            model_name='interview',
            name='duration',
        ),
    ]
