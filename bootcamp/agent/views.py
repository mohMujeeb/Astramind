import os
import json
import shutil
from pathlib import Path

from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
from django.shortcuts import render
from django.utils import timezone

from .models import Conversation, Message
from .memory import MemoryManager
from .agent_bridge import handle_chat  # supports override_index_dir in this setup
from src.ingest import run_ingest  # your existing ingest (now supports .pdf/.txt/.md)


# ---------------- Helpers ----------------

def _get_or_create_conversation(request):
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key
    conv, _ = Conversation.objects.get_or_create(session_key=session_key)
    return conv


BASE_DIR = Path(getattr(settings, "BASE_DIR"))
USER_DOCS_ROOT = BASE_DIR / "data" / "user_docs"
USER_INDEX_ROOT = BASE_DIR / "data" / "user_index"


def _session_paths(session_key: str):
    """
    Returns (docs_dir, index_dir) for this browser session.
    """
    return (
        USER_DOCS_ROOT / session_key,
        USER_INDEX_ROOT / session_key,
    )


def _render_user_history_markdown(messages):
    """
    Render a compact table of the last N user queries.
    """
    rows = []
    for i, m in enumerate(messages, start=1):
        ts = timezone.localtime(m.created_at).strftime("%Y-%m-%d %H:%M")
        content = (m.content or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {i} | {content} | {ts} |")
    if not rows:
        return "No history found."
    header = "| # | query | time |"
    sep = "| --- | --- | --- |"
    return "\n".join([header, sep, *rows])


# ---------------- Upload / Clear APIs ----------------

@method_decorator(csrf_exempt, name="dispatch")
class UploadAPI(View):
    """
    Accepts multiple .pdf / .txt / .md files, saves them under data/user_docs/<session>/,
    and builds a per-session FAISS index at data/user_index/<session>/ using run_ingest().
    """
    def post(self, request):
        conv = _get_or_create_conversation(request)
        docs_dir, index_dir = _session_paths(conv.session_key)

        files = request.FILES.getlist("files")
        if not files:
            return JsonResponse({"error": "no files uploaded"}, status=400)

        allowed = {".pdf", ".txt", ".md"}
        docs_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for f in files:
            ext = Path(f.name).suffix.lower()
            if ext not in allowed:
                continue
            safe_name = f.name.replace("/", "_").replace("\\", "_")
            out_path = docs_dir / safe_name
            with out_path.open("wb") as out:
                for chunk in f.chunks():
                    out.write(chunk)
            saved += 1

        if saved == 0:
            return JsonResponse({"error": "no valid .pdf/.txt/.md files"}, status=400)

        # Build (or rebuild) per-session index
        try:
            run_ingest(str(docs_dir), str(index_dir))
        except Exception as e:
            return JsonResponse({"error": f"ingest failed: {e}"}, status=500)

        return JsonResponse({"indexed": saved, "index_dir": str(index_dir)}, status=200)


@method_decorator(csrf_exempt, name="dispatch")
class ClearDocsAPI(View):
    """
    Removes both user docs and their per-session index.
    """
    def post(self, request):
        conv = _get_or_create_conversation(request)
        docs_dir, index_dir = _session_paths(conv.session_key)

        removed = 0
        for p in (docs_dir, index_dir):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1

        return JsonResponse({"removed_dirs": removed}, status=200)


# ---------------- Chat API ----------------

@method_decorator(csrf_exempt, name="dispatch")
class ChatAPI(View):
    def post(self, request):
        # Parse JSON body
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "invalid JSON body"}, status=400)

        user_text = (data.get("message") or "").strip()
        meta = data.get("meta") or {}

        if not user_text:
            return JsonResponse({"error": "message is required"}, status=400)

        # Conversation + memory
        conv = _get_or_create_conversation(request)
        mem = MemoryManager(user=request.user, session_key=conv.session_key, conversation=conv)

        # Persist user message
        Message.objects.create(conversation=conv, role="user", content=user_text)

        # Lightweight server-side "history" path (only last N user queries)
        if "history" in user_text.lower():
            limit = int(meta.get("historyLimit") or 10)
            recent = list(conv.messages.filter(role="user").order_by("-created_at")[:limit])
            final = _render_user_history_markdown(recent)
            Message.objects.create(conversation=conv, role="assistant", content=final, trace=None)
            return JsonResponse({"final": final, "trace": None, "rewritten": None}, status=200)

        # Per-session RAG index override + controller routing hint
        override_index_dir = None
        if bool(meta.get("useUserDocs")):
            _, idx_dir = _session_paths(conv.session_key)
            override_index_dir = str(idx_dir)
            # Prefix to trigger your controller's explicit RAG rule (see controller._route_part)
            user_text = f"According to our local docs, {user_text}"

        # Call agent
        try:
            # Preferred: handle_chat supports override_index_dir
            final, meta_out = handle_chat(user_text, mem, override_index_dir=override_index_dir)
        except TypeError:
            # Backward compatibility: old handle_chat signature → best-effort env var
            if override_index_dir:
                os.environ["ASTRAMIND_USER_INDEX"] = override_index_dir
            final, meta_out = handle_chat(user_text, mem)
            if override_index_dir:
                os.environ.pop("ASTRAMIND_USER_INDEX", None)

        # Persist assistant message
        Message.objects.create(
            conversation=conv,
            role="assistant",
            content=final,
            trace=(meta_out or {}).get("trace"),
        )

        return JsonResponse(
            {
                "final": final,
                "trace": (meta_out or {}).get("trace"),
                "rewritten": (meta_out or {}).get("rewritten"),
            },
            status=200,
        )


# ---------------- Page (empty chat on refresh) ----------------

def chat_page(request):
    """
    Always render a clean chat area to keep the UI tidy on refresh.
    Conversation still exists in DB for API/memory.
    """
    _get_or_create_conversation(request)
    return render(request, "chat/index.html", {"messages": []})
