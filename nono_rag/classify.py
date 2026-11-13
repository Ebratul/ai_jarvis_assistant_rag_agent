"""Rule-based enrichment: category, topics and entities for every instruction (English + Bangla).

Rules are deterministic and free (no LLM quota spent), so the brain can classify years of
instructions offline. The backend may override the category when it already knows the intent.
"""
from __future__ import annotations

import re
from collections import Counter

from .embeddings import normalize

# category -> (colour, keywords). Order matters: first match with the highest score wins.
CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "reminder": ("#f5a524", ["remind", "reminder", "alarm", "don't forget", "dont forget", "মনে করিয়ে", "মনে করাবে", "রিমাইন্ডার", "অ্যালার্ম", "এলার্ম"]),
    "calendar": ("#22d3ee", ["calendar", "meeting", "schedule", "event", "appointment", "agenda", "ক্যালেন্ডার", "মিটিং", "সময়সূচি", "শিডিউল"]),
    "telegram": ("#38bdf8", ["telegram", "message", "send", "text him", "text her", "dm", "টেলিগ্রাম", "মেসেজ", "পাঠাও", "পাঠিয়ে"]),
    "system": ("#a78bfa", ["switch", "model", "voice", "mute", "unmute", "volume", "louder", "settings", "মডেল", "ভয়েস", "কণ্ঠ"]),
    "memory": ("#34d399", ["remember", "note", "save this", "write down", "my favorite", "my favourite", "মনে রাখ", "নোট", "লিখে রাখ"]),
    "work": ("#60a5fa", ["project", "code", "deploy", "client", "office", "task", "deadline", "repo", "github", "bug", "কাজ", "অফিস", "প্রজেক্ট"]),
    "personal": ("#f472b6", ["family", "mom", "mother", "dad", "father", "wife", "husband", "friend", "birthday", "anniversary", "মা", "আম্মু", "বাবা", "আব্বু", "বন্ধু", "জন্মদিন", "বউ"]),
    "health": ("#4ade80", ["gym", "workout", "exercise", "medicine", "doctor", "water", "sleep", "diet", "ওষুধ", "ডাক্তার", "ব্যায়াম", "ঘুম", "পানি"]),
    "finance": ("#facc15", ["pay", "bill", "money", "budget", "bank", "salary", "invoice", "expense", "টাকা", "বিল", "বেতন", "খরচ"]),
    "learning": ("#c084fc", ["learn", "study", "course", "read", "book", "tutorial", "explain", "পড়", "শিখ", "বই"]),
    "web": ("#fb923c", ["search", "google", "weather", "news", "wikipedia", "look up", "আবহাওয়া", "খবর", "সার্চ"]),
    "question": ("#e5e7eb", []),
    "general": ("#94a3b8", []),
}

_INTENT_CATEGORIES = {"reminder", "calendar", "telegram", "system", "memory"}

_QUESTION_START = re.compile(
    r"^(what|who|whom|whose|why|how|when|where|which|is|are|can|could|do|does|did|will|would|should|কি|কী|কে|কেন|কিভাবে|কীভাবে|কখন|কোথায়|কোনটা)\b",
    re.IGNORECASE,
)

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    am i me my mine you your yours he him his she her it its we us our they them their this that these
    those there here what which who whom whose why how when where can could would should will shall may
    might must do does did done have has had having not no yes please nono hey hi hello ok okay just
    also very really so too some any all each every about into over under again more most less few
    tell give get make let lets let's want need like know think say said go going now today tomorrow
    me up out off down set call show find open use using one two three new time
    আমি আমার আমাকে তুমি তোমার আপনি আপনার সে তার এটা ওটা এই ওই কি কী কে কেন একটা একটি এবং ও বা
    না হ্যাঁ করো কর করে করবে করতে দাও দিন বলো বল হবে আছে ছিল জন্য থেকে দিয়ে নিয়ে সাথে আজ কাল
    মনে করিয়ে দিও দিবে রাখো রাখ কেমন আজকের এখন একটু প্লিজ তো যে টা টি""".split()
)

_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
_TIME = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s?(am|pm)|tomorrow|tonight|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|next month)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z'+#.-]{2,}|[ঀ-৿]{2,}")


def category_color(category: str) -> str:
    return CATEGORIES.get(category, CATEGORIES["general"])[0]


def classify(text: str) -> str:
    t = normalize(text)
    scores: Counter[str] = Counter()
    for cat, (_, keywords) in CATEGORIES.items():
        weight = 3 if cat in _INTENT_CATEGORIES else 1  # actions outrank subject matter
        for kw in keywords:
            if kw in t:
                scores[cat] += weight * (2 if " " in kw else 1)
    if scores:
        return scores.most_common(1)[0][0]
    if t.endswith("?") or _QUESTION_START.match(t):
        return "question"
    return "general"


def topics(text: str, limit: int = 5) -> list[str]:
    seen: list[str] = []
    for tok in _TOKEN.findall(text or ""):
        w = tok.lower().strip(".'-")
        if len(w) < 3 or w in STOPWORDS or w.isdigit() or w in seen:
            continue
        seen.append(w)
    # Prefer longer, more specific words while keeping original order stable.
    ranked = sorted(seen, key=lambda w: (-min(len(w), 9), seen.index(w)))[:limit]
    return [w for w in seen if w in ranked]


def entities(text: str) -> list[str]:
    found: list[str] = []
    for m in _ENTITY.finditer(text or ""):
        e = m.group(1).strip()
        if e.lower() not in STOPWORDS and e not in found and e.lower() != "nono":
            found.append(e)
    for m in _TIME.finditer(text or ""):
        e = m.group(0).lower()
        if e not in found:
            found.append(e)
    return found[:8]


def importance(text: str, category: str) -> float:
    score = 0.4
    if category in {"reminder", "calendar", "memory", "personal", "finance", "health"}:
        score += 0.25
    if any(w in normalize(text) for w in ("important", "always", "never", "must", "জরুরি", "সবসময়")):
        score += 0.25
    return round(min(score, 1.0), 2)
