from .views import UserViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'user-view', UserViewSet, basename='user')
urlpatterns = router.urls
