# Generated migration for adding evaluation_results field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0007_completedinterview_summary'),
    ]

    operations = [
        migrations.AddField(
            model_name='completedinterview',
            name='evaluation_results',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
