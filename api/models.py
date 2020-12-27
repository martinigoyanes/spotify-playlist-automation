from django.db import models

# Create your models here.
class User(models.Model):
    acces_token = models.CharField(max_length=150, default="", unique=True)
    refresh_token = models.CharField(max_length=150, default="", unique=True)
    expires_in = models.DateTimeField()
    trained_model = models.CharField(max_length=150, default="", unique=True)
    uid = models.CharField(max_length=150, default="", unique=True)
