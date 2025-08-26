import re
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from django.utils import timezone
from .models import MemoryItem, Conversation, Message

NUM_RX = re.compile(r"-?\d+(?:\.\d+)?")

class MemoryManager:
    def __init__(self, user=None, session_key: Optional[str]=None, conversation: Optional[Conversation]=None):
        self.user = user if getattr(user, "is_authenticated", False) else None
        self.session_key = None if self.user else (session_key or None)
        self.conversation = conversation

    # ---------- persistence ----------

    def _upsert(self, scope: str, namespace: str, key: str,
                data_type: str, value_text: str = "", number_value: Decimal | None = None,
                value_json: Dict[str, Any] | None = None, persistent: bool = True, src_msg=None):
        mi, _ = MemoryItem.objects.update_or_create(
            user=self.user, session_key=self.session_key, conversation=self.conversation,
            scope=scope, namespace=namespace, key=key,
            defaults={
                "data_type": data_type,
                "value_text": value_text,
                "number_value": number_value,
                "value_json": value_json,
                "is_persistent": persistent,
                "source_message": src_msg,
            }
        )
        return mi

    def get(self, scope: str, namespace: str, key: str) -> Optional[MemoryItem]:
        return MemoryItem.objects.filter(
            user=self.user, session_key=self.session_key, conversation=self.conversation,
            scope=scope, namespace=namespace, key=key
        ).first()

    # ---------- numeric memory ----------

    def set_last_number(self, x, tool: str | None = None, src_msg=None):
        dec = Decimal(str(x))
        self._upsert("thread", "numeric", "last", "number", number_value=dec, src_msg=src_msg, persistent=False)
        if tool:
            self._upsert("thread", "numeric", f"{tool}.last", "number", number_value=dec, src_msg=src_msg, persistent=False)

    def get_last_number(self, prefer_tool: str | None = None):
        if prefer_tool:
            item = self.get("thread", "numeric", f"{prefer_tool}.last")
            if item and item.number_value is not None:
                return item.number_value
        item = self.get("thread", "numeric", "last")
        return item.number_value if (item and item.number_value is not None) else None

    # Fallback source for a number when user says "add 70" but numeric.last is empty.
    # Tries: last assistant message number → last stored fact number.
    def get_best_followup_number(self) -> Optional[Decimal]:
        # 1) thread last number
        n = self.get_last_number()
        if n is not None:
            return n
        # 2) last assistant message number
        if self.conversation:
            last_assistant = Message.objects.filter(conversation=self.conversation, role="assistant").order_by("-created_at").first()
            if last_assistant:
                nums = NUM_RX.findall(last_assistant.content or "")
                if nums:
                    try:
                        return Decimal(nums[-1])
                    except Exception:
                        pass
        # 3) last durable fact value with a number
        items = MemoryItem.objects.filter(
            user=self.user, session_key=self.session_key, conversation=self.conversation,
            scope__in=["user", "session"], namespace__in=["web.fact", "rag.fact"]
        ).order_by("-updated_at")[:20]
        for it in items:
            # look inside text/json for a numeric
            text = it.value_text or ""
            nums = NUM_RX.findall(text)
            if nums:
                try:
                    return Decimal(nums[-1])
                except Exception:
                    continue
        return None

    # ---------- entity tracking ----------

    def set_last_entity(self, etype: str, name: str, persistent: bool=False, src_msg=None):
        self._upsert("thread", "entity", f"{etype}.last", "text", value_text=name, persistent=not persistent, src_msg=src_msg)

    def get_last_entity(self, etype: str) -> Optional[str]:
        it = self.get("thread", "entity", f"{etype}.last")
        return it.value_text if it else None

    # ---------- facts ----------

    def set_fact(self, namespace: str, key: str, value: str, src_msg=None, from_tool: str="web"):
        self._upsert("user" if self.user else "session", f"{namespace}", key, "text",
                     value_text=value, persistent=True, src_msg=src_msg)

    def get_fact(self, namespace: str, key: str) -> Optional[str]:
        it = self.get("user" if self.user else "session", f"{namespace}", key)
        return it.value_text if it else None

    # ---------- capture helpers ----------

    def capture_numbers_from_text(self, text: str, prefer_tool: str | None = None, src_msg=None):
        if not text: return
        nums = NUM_RX.findall(text)
        if nums:
            try:
                self.set_last_number(nums[-1], tool=prefer_tool, src_msg=src_msg)
            except Exception:
                pass

    def capture_topic_from_query(self, query: str):
        m = re.search(r"\bcapital of ([A-Z][A-Za-z ]+)\b", query, re.I)
        if m: self.set_last_entity("country", m.group(1).strip())
        m = re.search(r"\bceo of ([A-Z0-9][\w .&-]+)\b", query, re.I)
        if m: self.set_last_entity("company", m.group(1).strip())
        m = re.search(r"\bheight of ([A-Z][\w -]*Mount|\bMount [A-Z][\w -]*)\b|\bheight of ([A-Z][\w -]+)\b", query, re.I)
        if m: self.set_last_entity("mountain", (m.group(1) or m.group(2)).strip())
        m = re.search(r"\bpopulation of ([A-Z][A-Za-z ]+)\b", query, re.I)
        if m: self.set_last_entity("country", m.group(1).strip())

    def maybe_store_fact_from_qa(self, question: str, answer: str, tool: str, src_msg=None):
        m = re.search(r"\bcapital of ([A-Z][A-Za-z ]+)\b", question, re.I)
        if m:
            country = m.group(1).strip()
            city = self._extract_proper_phrase(answer) or answer.strip()
            self.set_fact("web.fact" if tool == "web" else "rag.fact", f"capital:{country}", city, src_msg=src_msg, from_tool=tool)

        m = re.search(r"\bceo of ([A-Z0-9][\w .&-]+)\b", question, re.I)
        if m:
            company = m.group(1).strip()
            person = self._extract_proper_phrase(answer) or answer.strip()
            self.set_fact("web.fact", f"ceo:{company}", person, src_msg=src_msg, from_tool=tool)

        m = re.search(r"\bheight of ([A-Z][\w -]+)\b", question, re.I)
        if m:
            obj = m.group(1).strip()
            self.set_fact("rag.fact" if tool == "rag" else "web.fact", f"height:{obj}", answer.strip(), src_msg=src_msg, from_tool=tool)

    @staticmethod
    def _extract_proper_phrase(text: str) -> Optional[str]:
        m = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", text)
        return m.group(1).strip() if m else None

    # ---------- follow-up rewriting (robust) ----------

    ADD_RX = re.compile(r"\b(add|plus|\+)\s*(-?\d+(?:\.\d+)?)\b", re.I)
    SUB_RX = re.compile(r"\b(subtract|minus|\-)\s*(-?\d+(?:\.\d+)?)\b", re.I)
    MUL_RX = re.compile(r"\b(multiply|times|\*)\s*(?:by\s*)?(-?\d+(?:\.\d+)?)\b", re.I)
    DIV_RX = re.compile(r"\b(divide|\/)\s*(?:by\s*)?(-?\d+(?:\.\d+)?)\b", re.I)
    IT_RX  = re.compile(r"\b(it|that|the result|this number)\b", re.I)

    PLUS_RX   = re.compile(r"\bplus\s+(-?\d+(?:\.\d+)?)\b", re.I)
    INCR_RX   = re.compile(r"\b(increase|increment|add)\s*(?:it|that|the result|this number)?\s*(?:by\s*)?(-?\d+(?:\.\d+)?)\b", re.I)
    DECR_RX   = re.compile(r"\b(decrease|reduce)\s*(?:it|that|the result|this number)?\s*(?:by\s*)?(-?\d+(?:\.\d+)?)\b", re.I)
    TIMES_RX  = re.compile(r"\b(?:x|times)\s*(-?\d+(?:\.\d+)?)\b", re.I)
    OVER_RX   = re.compile(r"\b(?:over)\s*(-?\d+(?:\.\d+)?)\b", re.I)
    FULL_EXPR_RX = re.compile(r"\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?")

    def rewrite_numeric_followup(self, user_text: str, prefer_tool: str | None = None) -> Optional[str]:
        """
        Rewrites 'add 70', 'plus 5', 'decrease by 3', 'x 4', 'over 2', etc. to '<last><op><n>'.
        Works without pronouns. Falls back to last assistant numeric or last fact numeric.
        """
        last = (self.get_last_number(prefer_tool=prefer_tool) or
                self.get_last_number() or
                self.get_best_followup_number())
        if last is None:
            return None

        t = user_text.strip()
        # If user already typed a math expression, don't rewrite.
        if self.FULL_EXPR_RX.search(t) or (re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", t) and re.search(r"[\+\-\*\/]", t)):
            return None

        patterns = [
            (self.ADD_RX, "+"), (self.PLUS_RX, "+"), (self.INCR_RX, "+"),
            (self.SUB_RX, "-"), (self.DECR_RX, "-"),
            (self.MUL_RX, "*"), (self.TIMES_RX, "*"),
            (self.DIV_RX, "/"), (self.OVER_RX, "/"),
        ]
        op = None
        n_val: Optional[Decimal] = None
        for rx, sym in patterns:
            m = rx.search(t)
            if m:
                try:
                    n_val = Decimal(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
                except Exception:
                    continue
                op = sym
                break

        if not op or n_val is None:
            return None

        nums = NUM_RX.findall(t)
        if len(nums) > 1:
            # too ambiguous (e.g., "add 2 and 3"), don't rewrite
            return None

        L = last.quantize(Decimal("1")) if last == last.to_integral() else last
        R = n_val.quantize(Decimal("1")) if n_val == n_val.to_integral() else n_val
        return f"{L}{op}{R}"

    def rewrite_fact_followup(self, user_text: str) -> Optional[str]:
        text = user_text.lower()
        if "population" in text:
            subj = self.get_last_entity("country")
            if subj: return f"population of {subj}"
        if "capital" in text:
            subj = self.get_last_entity("country")
            if subj: return f"what is the capital of {subj}"
        if "ceo" in text:
            subj = self.get_last_entity("company")
            if subj: return f"who is the CEO of {subj}"
        if "height" in text:
            subj = self.get_last_entity("mountain")
            if subj: return f"what is the height of {subj}"
        return None

    # ---------- history / listing / clear ----------

    def recent_messages(self, limit: int = 10) -> List[Tuple[str, str, str]]:
        if not self.conversation:
            return []
        msgs = Message.objects.filter(conversation=self.conversation).order_by("-created_at")[:limit]
        out = []
        for m in reversed(list(msgs)):
            out.append((m.role, m.content, m.created_at.astimezone(timezone.get_current_timezone()).strftime("%Y-%m-%d %H:%M")))
        return out

    def recent_user_questions(self, limit: int = 10) -> List[Tuple[str, str]]:
        if not self.conversation:
            return []
        msgs = Message.objects.filter(conversation=self.conversation, role="user").order_by("-created_at")[:limit]
        return [(m.content, m.created_at.astimezone(timezone.get_current_timezone()).strftime("%Y-%m-%d %H:%M")) for m in reversed(list(msgs))]

    def list_stored_facts(self) -> List[Tuple[str, str, str]]:
        # returns (namespace, key, value_text)
        items = MemoryItem.objects.filter(
            user=self.user, session_key=self.session_key, conversation=self.conversation,
            scope__in=["user", "session"], namespace__in=["web.fact", "rag.fact"]
        ).order_by("namespace", "key")
        return [(it.namespace, it.key, it.value_text or "") for it in items]

    def clear_thread_memory(self) -> int:
        qs = MemoryItem.objects.filter(
            user=self.user, session_key=self.session_key, conversation=self.conversation,
            scope="thread"
        )
        n = qs.count()
        qs.delete()
        return n
