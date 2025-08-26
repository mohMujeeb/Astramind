from django.db import models
from django.contrib.auth import get_user_model
from django.db.models import Q

User = get_user_model()

class Conversation(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    session_key = models.CharField(max_length=64, null=True, blank=True, db_index=True)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=[("user","user"),("assistant","assistant")])
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    trace = models.JSONField(null=True, blank=True)

class ToolRun(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="tool_runs")
    message = models.ForeignKey(Message, null=True, blank=True, on_delete=models.SET_NULL)
    step_id = models.CharField(max_length=40)
    tool_name = models.CharField(max_length=32)
    input_text = models.TextField()
    output_text = models.TextField()
    meta = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class MemoryItem(models.Model):
    SCOPE_CHOICES = [("thread","thread"), ("user","user"), ("session","session")]
    TYPE_CHOICES  = [("number","number"), ("text","text"), ("json","json")]

    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    session_key = models.CharField(max_length=64, null=True, blank=True, db_index=True)
    conversation = models.ForeignKey(Conversation, null=True, blank=True, on_delete=models.CASCADE)

    scope = models.CharField(max_length=10, choices=SCOPE_CHOICES, default="thread")
    namespace = models.CharField(max_length=48, db_index=True)
    key = models.CharField(max_length=120, db_index=True)

    data_type = models.CharField(max_length=8, choices=TYPE_CHOICES, default="text")
    value_text = models.TextField(blank=True, default="")
    number_value = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    value_json = models.JSONField(null=True, blank=True)

    is_persistent = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    source_message = models.ForeignKey(Message, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["namespace", "key"]),
            models.Index(fields=["scope", "namespace", "key"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "scope", "namespace", "key"],
                name="uniq_user_mem_per_key",
                condition=Q(user__isnull=False),
            ),
            models.UniqueConstraint(
                fields=["session_key", "scope", "namespace", "key"],
                name="uniq_session_mem_per_key",
                condition=Q(user__isnull=True, session_key__isnull=False),
            ),
            models.UniqueConstraint(
                fields=["conversation", "scope", "namespace", "key"],
                name="uniq_thread_mem_per_key",
                condition=Q(conversation__isnull=False),
            ),
        ]
