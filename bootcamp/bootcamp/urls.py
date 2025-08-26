# bootcamp/urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("agent/", include("agent.urls")),
    path("", TemplateView.as_view(template_name="chat/landing.html"), name="home"),      # landing
    path("learn/", TemplateView.as_view(template_name="chat/about.html"), name="learn"), # docs
]
