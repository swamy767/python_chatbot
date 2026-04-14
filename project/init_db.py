import sqlite3
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'faq.db')
JSON_PATH = os.path.join(BASE_DIR, 'intents.json')

def init_db():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop table if it exists to ensure fresh migration
    cursor.execute("DROP TABLE IF EXISTS intents")

    # Create table
    cursor.execute("""
        CREATE TABLE intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE NOT NULL,
            patterns TEXT NOT NULL,
            responses TEXT NOT NULL
        )
    """)
    print("Created 'intents' table schema.")

    print(f"Reading data from {JSON_PATH}...")
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {JSON_PATH} not found!")
        return

    # Insert data
    insert_query = "INSERT INTO intents (tag, patterns, responses) VALUES (?, ?, ?)"
    
    count = 0
    for intent in data.get('intents', []):
        tag = intent.get('tag', '')
        patterns = json.dumps(intent.get('patterns', []))
        responses = json.dumps(intent.get('responses', []))
        
        cursor.execute(insert_query, (tag, patterns, responses))
        count += 1

    conn.commit()
    conn.close()
    
    print(f"Successfully migrated {count} intents into faq.db SQLite database!")

if __name__ == "__main__":
    init_db()
