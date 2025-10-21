# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0002_interview'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='interview',
            name='duration',
        ),
        migrations.RemoveField(
            model_name='interview',
            name='interview_type',
        ),
        migrations.AddField(
            model_name='interview',
            name='industry',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='interview',
            name='role',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
