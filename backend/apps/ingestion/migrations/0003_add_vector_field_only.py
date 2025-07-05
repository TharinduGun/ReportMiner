# Generated migration for adding vector field only
from django.db import migrations
from pgvector.django import VectorField

class Migration(migrations.Migration):

    dependencies = [
        ('ingestion', '0002_add_vector_field'),
    ]

    operations = [
        migrations.AddField(
            model_name='documenttextsegment',
            name='embedding',
            field=VectorField(dimensions=1536, null=True, blank=True),
        ),
    ]