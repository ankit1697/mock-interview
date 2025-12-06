from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from backend.views import direct_google_login

urlpatterns = [
    path('admin/', admin.site.urls),
    # Place direct Google login before allauth URLs to take precedence
    path('accounts/google/login/direct/', direct_google_login, name='direct_google_login'),
    path('accounts/', include('allauth.urls')),  # Add this line
    path('', include('backend.urls')),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)