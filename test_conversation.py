import requests
import json
import time

import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = "" # Set if needed, but bot.py shows it's optional if API_ACCESS_KEY is not set

def test_new_format_text():
    print("Testing New Format (Text)...")
    payload = {
        "session_id": "test-session-text",
        "input": {
            "type": "text",
            "content": "Salom, qandaysiz?"
        },
        "context": {
            "msisdn": "+998901234567",
            "platform": "telegram",
            "language": "uz"
        }
    }
    resp = requests.post(f"{BASE_URL}/conversation", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "test-session-text"

def test_new_format_voice():
    print("\nTesting New Format (Voice)...")
    payload = {
        "session_id": "test-session-voice",
        "input": {
            "type": "voice",
            "content": "Bu ovozli habar matni"
        },
        "context": {
            "msisdn": "+998901234567",
            "platform": "telegram",
            "language": "uz"
        }
    }
    resp = requests.post(f"{BASE_URL}/conversation", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "test-session-voice"

def test_legacy_format():
    print("\nTesting Legacy Format...")
    payload = {
        "message": "Legacy message content"
    }
    resp = requests.post(f"{BASE_URL}/conversation", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert "legacy-" in resp.json()["session_id"]

def test_recent():
    print("\nTesting /recent...")
    resp = requests.get(f"{BASE_URL}/recent")
    print(f"Status: {resp.status_code}")
    items = resp.json()["items"]
    print(f"Last item: {items[-1] if items else 'None'}")
    assert len(items) >= 3

if __name__ == "__main__":
    try:
        test_new_format_text()
        test_new_format_voice()
        test_legacy_format()
        test_recent()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
