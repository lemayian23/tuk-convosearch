"""
Test the chat API
Location: backend/test_api.py
"""

import requests
import json

# API endpoint
url = "http://localhost:8000/api/chat"

# Question to ask
question = "When do exams start?"

# Send request
response = requests.post(
    url,
    json={"message": question, "session_id": "test"}
)

if response.status_code == 200:
    result = response.json()
    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print("ANSWER:")
    print(result['answer'])
    print("\n" + "=" * 60)
    print("SOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"{i+1}. {source['source']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)