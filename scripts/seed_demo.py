"""Seed a *separate* demo brain with realistic instructions spread over several years.

Usage:  python scripts/seed_demo.py --url http://localhost:8100 --count 220
Every seeded memory has source="demo" so it can be filtered or removed later.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

import httpx

TEMPLATES = [
    "Remind me to call {person} at {time}", "Remind me to take my {med} after dinner",
    "Schedule a meeting with {person} on {day} about the {project} project",
    "Send a telegram message to {person}: running 10 minutes late",
    "Switch the model to {model}", "Change your voice to {voice}",
    "Remember that my favourite {thing} is {value}", "Note: {project} deadline is next {day}",
    "What is the weather in {city} today?", "Search the news about {topic}",
    "Explain {topic} in simple words", "How do I learn {topic} fast?",
    "Pay the {bill} bill before {day}", "Track my budget for {thing}",
    "Start a workout plan for the gym", "Drink water reminder every two hours",
    "Deploy the {project} repo to production", "Fix the login bug in {project}",
    "Book a doctor appointment for {day}", "Wish {person} a happy birthday on telegram",
    "আমাকে {time} এ মনে করিয়ে দিও", "আজকের আবহাওয়া কেমন {city} এ?", "মডেল পরিবর্তন করো {model}",
    "আম্মুকে ফোন করতে মনে করিয়ে দিও", "আমার প্রিয় খাবার বিরিয়ানি মনে রাখো",
]
FILL = {
    "person": ["Mom", "Dad", "Rahim", "Karim", "Ayesha", "Sadia", "Tanvir", "the team"],
    "time": ["9am", "5pm", "8:30pm", "noon", "7am"],
    "med": ["vitamin D", "medicine", "omega 3"],
    "day": ["Monday", "Tuesday", "Friday", "Sunday"],
    "project": ["Nono", "Phoenix", "Atlas", "website", "mobile app"],
    "model": ["groq llama", "gemma", "nemotron", "gpt", "openrouter free"],
    "voice": ["Aria", "Titan", "Nova", "Echo", "Nono Prime"],
    "thing": ["color", "coffee", "travel", "food", "books"],
    "value": ["blue", "black coffee", "Sylhet", "biryani", "sci-fi"],
    "city": ["Dhaka", "Chittagong", "Sylhet", "London", "Tokyo"],
    "topic": ["quantum computing", "machine learning", "rust", "startups", "neural networks", "investing"],
    "bill": ["internet", "electricity", "credit card"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--count", type=int, default=220)
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--key", default="")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    headers = {"X-Nono-Key": args.key} if args.key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as c:
        for _ in range(args.count):
            tpl = random.choice(TEMPLATES)
            text = tpl.format(**{k: random.choice(v) for k, v in FILL.items()})
            when = now - timedelta(days=random.random() ** 1.6 * 365 * args.years, minutes=random.randint(0, 1440))
            c.post("/memories", json={"text": text, "source": "demo", "created_at": when.isoformat(timespec="seconds"),
                                     "response": "Done." if random.random() < 0.6 else None}).raise_for_status()
    print(f"Seeded {args.count} demo memories into {args.url}")


if __name__ == "__main__":
    main()
