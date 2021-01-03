# Generated by Django 3.1.4 on 2021-01-02 23:26

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AIConfig',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.CharField(max_length=50, unique=True)),
                ('maxlen', models.IntegerField()),
                ('vocab_size', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='AIEncoding',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.CharField(max_length=50)),
                ('spotify_id', models.CharField(max_length=50, unique=True)),
                ('playlist_name', models.CharField(max_length=50)),
                ('encoding', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='PlaylistModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.CharField(max_length=50)),
                ('spotify_id', models.CharField(max_length=50, unique=True)),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uid', models.CharField(max_length=50, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('refresh_token', models.CharField(max_length=150)),
                ('access_token', models.CharField(max_length=150)),
                ('expires_in', models.DateTimeField()),
                ('token_type', models.CharField(max_length=50)),
                ('curr_session', models.CharField(max_length=50, unique=True)),
                ('model', models.FileField(blank=True, upload_to='')),
            ],
        ),
    ]
