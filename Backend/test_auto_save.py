#!/usr/bin/env python3
"""
Test script to verify auto-save functionality.
This simulates what happens when you send a query to the backend.
"""
import requests
import time
import uuid

# Test the auto-save functionality
def test_auto_save():
    base_url = "http://localhost:8000"

    # Create a test session
    session_id = str(uuid.uuid4())
    print(f"🧪 Testing with session_id: {session_id}")

    # Send a test query
    payload = {
        "query": "Hello, this is a test message",
        "session_id": session_id
    }

    try:
        print("📤 Sending test query...")
        response = requests.post(f"{base_url}/query", json=payload)
        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful, got session_id: {data.get('session_id')}")

            # Wait a moment for background save to complete
            print("⏳ Waiting for auto-save to complete...")
            time.sleep(2)

            # Check if conversation was created in neo4j
            try:
                from neo4j_auth import _run
                convs = _run(
                    "MATCH (c:Conversation {session_id: $sid}) RETURN c.id, c.title",
                    {"sid": session_id}
                )
                if convs:
                    conv_id = convs[0]["c.id"]
                    print(f"✅ Conversation created: {conv_id}")

                    # Check messages
                    msgs = _run(
                        "MATCH (:Conversation {id: $cid})-[r:CONTAINS]->(m:Message) RETURN m.role, m.content ORDER BY r.index",
                        {"cid": conv_id}
                    )
                    print(f"✅ Messages saved: {len(msgs)}")
                    for msg in msgs:
                        print(f"   {msg['m.role']}: {msg['m.content'][:50]}...")
                else:
                    print("❌ No conversation found in neo4j")
            except Exception as e:
                print(f"❌ Neo4j check failed: {e}")
        else:
            print(f"❌ Query failed: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_auto_save()