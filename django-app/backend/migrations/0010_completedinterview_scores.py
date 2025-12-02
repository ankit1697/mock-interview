from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("backend", "0009_remove_interview_interview_type_and_duration"),
    ]

    operations = [
        migrations.AddField(
            model_name="completedinterview",
            name="overall_score",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="completedinterview",
            name="technical_reasoning_avg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="completedinterview",
            name="accuracy_avg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="completedinterview",
            name="confidence_avg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="completedinterview",
            name="problem_solving_avg",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="completedinterview",
            name="flow_avg",
            field=models.FloatField(null=True, blank=True),
        ),
    ]


