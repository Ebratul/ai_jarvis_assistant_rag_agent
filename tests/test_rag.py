import json

from fastapi.testclient import TestClient

from nono_rag import classify
from nono_rag.main import app


def test_classifier_english_and_bangla():
    assert classify.classify("Remind me to call mom at 5pm") == "reminder"
    assert classify.classify("আমাকে কাল সকালে মনে করিয়ে দিও") == "reminder"
    assert classify.classify("Schedule a meeting with Rahim on Monday") == "calendar"
    assert classify.classify("send a telegram message to my team") == "telegram"
    assert classify.classify("what is quantum computing?") in {"question", "learning"}
    assert "quantum" in classify.topics("what is quantum computing")
    assert "Rahim" in classify.entities("Schedule a meeting with Rahim on Monday")


def test_full_memory_lifecycle():
    with TestClient(app) as c:
        assert c.get("/health").json()["ok"]
        a = c.post("/memories", json={"text": "Remind me to drink water every 2 hours", "source": "voice"}).json()
        b = c.post("/memories", json={"text": "Remind me to drink more water after the gym", "created_at": "2031-05-01T08:00:00+00:00"}).json()
        c.post("/memories", json={"text": "Switch the model to groq llama"})
        c.post("/memories", json={"text": "আমার প্রিয় রং নীল মনে রাখো"})
        assert a["category"] == "reminder" and a["source"] == "voice"

        p = c.patch(f"/memories/{a['id']}", json={"response": "Done, reminder set.", "tools": ["set_reminder"], "pinned": True}).json()
        assert p["response"] == "Done, reminder set." and p["pinned"] is True

        hits = c.post("/search", json={"query": "drink water"}).json()["results"]
        assert hits and hits[0]["memory"]["id"] in {a["id"], b["id"]}
        bn = c.post("/search", json={"query": "প্রিয় রং"}).json()["results"]
        assert bn and "নীল" in bn[0]["memory"]["text"]

        listed = c.get("/memories", params={"q": "water"}).json()
        assert len(listed["items"]) == 2
        assert len(c.get("/memories", params={"year_from": 2031, "year_to": 2031}).json()["items"]) == 1

        detail = c.get(f"/memories/{a['id']}").json()
        assert detail["related"][0]["memory"]["id"] == b["id"]

        g = c.get("/graph", params={"similarity": 0.2}).json()
        types = g["counts"]
        assert types["core"] == 1 and types["memory"] == 4 and types["year"] == 2
        assert any(l["type"] == "similar" for l in g["links"])
        node_ids = {n["id"] for n in g["nodes"]}
        assert all(l["source"] in node_ids and l["target"] in node_ids for l in g["links"])

        s = c.get("/stats").json()
        assert s["total"] == 4 and s["by_category"]["reminder"] == 2

        exported = [json.loads(l) for l in c.get("/export").text.splitlines()]
        assert len(exported) == 4
        assert c.delete(f"/memories/{b['id']}").json()["deleted"]
        assert c.post("/import", content="\n".join(json.dumps(e) for e in exported)).json()["imported"] == 1
        assert c.get("/stats").json()["total"] == 4
        assert c.post("/admin/backup").json()["backup"].endswith(".db")
