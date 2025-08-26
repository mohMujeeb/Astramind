# agent/urls.py
from django.urls import path
from .views import ChatAPI, chat_page, UploadAPI, ClearDocsAPI

urlpatterns = [
    path("", chat_page, name="chat_page"),
    path("api/", ChatAPI.as_view(), name="chat_api"),
    path("upload/", UploadAPI.as_view(), name="upload_api"),
    path("clear-docs/", ClearDocsAPI.as_view(), name="clear_docs_api"),
]
