from .views import UserViewSet, SongModelViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'user-view', UserViewSet, basename='user')
router.register(r'songs-view', SongModelViewSet, basename='songs')
urlpatterns = router.urls
