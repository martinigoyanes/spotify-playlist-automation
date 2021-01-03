from rest_framework import serializers
from backend.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('uid', 'created_at', 'refresh_token', 'access_token', 'expires_in', 'token_type', 'model')